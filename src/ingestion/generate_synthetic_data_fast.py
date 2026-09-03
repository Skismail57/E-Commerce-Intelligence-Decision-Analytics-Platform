from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import random


class FastOrderGenerator:
    def __init__(
        self, customers_df, products_df, stores_df, campaigns_df,
        start_date, end_date, num_orders, random_state=42
    ):
        self.customers_df = customers_df.copy()
        self.products_df = products_df.copy()
        self.stores_df = stores_df.copy()
        self.campaigns_df = campaigns_df.copy()
        self.start_date = start_date
        self.end_date = end_date
        self.num_orders = int(num_orders)
        self.random_state = random_state

        self.active_products = self.products_df[
            self.products_df["product_status"] == "Active"
        ]["product_id"].to_numpy()

        self.store_ids = self.stores_df["store_id"].to_numpy()

        self.campaign_ids = self.campaigns_df["campaign_id"].to_numpy()
        self.campaign_starts = self.campaigns_df["start_date"].to_numpy()
        self.campaign_ends = self.campaigns_df["end_date"].to_numpy()

        self.cust_ids = self.customers_df["customer_id"].to_numpy()
        self.cust_signup_dates = self.customers_df["signup_date"].to_numpy()
        self.cust_segments = self.customers_df["customer_segment"].to_numpy()

        seg_sort_idx = np.argsort(self.cust_signup_dates)
        self.cust_ids_sorted = self.cust_ids[seg_sort_idx]
        self.cust_signup_sorted = self.cust_signup_dates[seg_sort_idx]
        self.cust_segments_sorted = self.cust_segments[seg_sort_idx]

        self.seg_weight_map = {
            "Champion": 0.25,
            "Loyal Customer": 0.20,
            "Potential Loyalist": 0.18,
            "New Customer": 0.15,
            "At Risk": 0.08,
            "Can't Lose Them": 0.07,
            "Lost Customer": 0.07,
        }
        self.cust_seg_weights = np.array([
            self.seg_weight_map.get(seg, 0.1) for seg in self.cust_segments_sorted
        ])

        signup_order = np.arange(len(self.cust_ids_sorted), dtype=np.float64)
        norm_factor = max(1.0, len(self.cust_ids_sorted))
        signup_order_weight = 1.0 / (1.0 + signup_order / norm_factor)
        self.cust_combined_weights = self.cust_seg_weights * signup_order_weight

        product_ids_arr = self.products_df["product_id"].to_numpy()
        selling_prices_arr = self.products_df["selling_price"].to_numpy()
        self.selling_price_map = dict(zip(product_ids_arr, selling_prices_arr))
        self._max_pid = int(product_ids_arr.max()) if len(product_ids_arr) > 0 else 0
        self._price_lookup = np.zeros(self._max_pid + 1, dtype=np.float64)
        self._price_lookup[product_ids_arr.astype(np.int64)] = selling_prices_arr

        self.total_days = (self.end_date - self.start_date).days + 1
        self.daily_dates = np.array([
            self.start_date + timedelta(days=i)
            for i in range(self.total_days)
        ], dtype=object)

        self.dt_daily_dates = pd.to_datetime(self.daily_dates)

        self._precompute_holiday_weights()

    def _precompute_holiday_weights(self):
        hw = np.ones(self.total_days, dtype=np.float64)

        daily_dt = self.dt_daily_dates
        daily_years = daily_dt.year.to_numpy()
        daily_months = daily_dt.month.to_numpy()
        daily_days = daily_dt.day.to_numpy()
        daily_weekdays = np.array([d.weekday() for d in self.daily_dates], dtype=np.int32)
        daily_ordinals = np.array([d.toordinal() for d in self.daily_dates], dtype=np.int64)

        diwali_dates = [
            (date(2022, 10, 24), 4.0),
            (date(2023, 11, 12), 4.0),
            (date(2024, 11, 1), 4.0),
        ]
        for ddate, peak in diwali_dates:
            diffs = daily_ordinals - ddate.toordinal()
            in_window = (diffs >= -10) & (diffs <= 5)
            if in_window.any():
                w_diff = diffs[in_window].astype(np.float64)
                mult = np.ones_like(w_diff)
                before = w_diff < 0
                t_before = (w_diff[before] + 10.0) / 10.0
                mult[before] = 1.5 + (peak - 1.5) * t_before
                at_day = w_diff == 0
                mult[at_day] = peak
                after = w_diff > 0
                t_after = w_diff[after] / 5.0
                mult[after] = 2.0 - 1.0 * t_after
                mult = np.maximum(mult, 1.0)
                hw[in_window] *= mult

        dussehra_dates = [
            (date(2022, 10, 5), 2.5),
            (date(2023, 10, 24), 2.5),
            (date(2024, 10, 12), 2.5),
        ]
        for ddate, peak in dussehra_dates:
            diffs = daily_ordinals - ddate.toordinal()
            in_window = (diffs >= -5) & (diffs <= 0)
            if in_window.any():
                w_diff = diffs[in_window].astype(np.float64)
                mult = np.ones_like(w_diff)
                before = w_diff < 0
                t_before = (w_diff[before] + 5.0) / 5.0
                mult[before] = 1.0 + (peak - 1.0) * t_before
                at_day = w_diff == 0
                mult[at_day] = peak
                hw[in_window] *= mult

        holi_ordinals = np.array([
            date(2022, 3, 18).toordinal(),
            date(2023, 3, 8).toordinal(),
            date(2024, 3, 25).toordinal(),
        ], dtype=np.int64)
        for ho in holi_ordinals:
            mask = daily_ordinals == ho
            hw[mask] *= 2.0

        for year in [2022, 2023, 2024]:
            xmas = date(year, 12, 25).toordinal()
            mask = daily_ordinals == xmas
            hw[mask] *= 1.8

        for year in [2022, 2023, 2024]:
            ny1 = date(year, 12, 31).toordinal()
            ny2 = date(year + 1, 1, 1).toordinal()
            mask = (daily_ordinals == ny1) | (daily_ordinals == ny2)
            hw[mask] *= 1.7

        for year in [2022, 2023, 2024]:
            rep = date(year, 1, 26).toordinal()
            ind = date(year, 8, 15).toordinal()
            mask = (daily_ordinals == rep) | (daily_ordinals == ind)
            hw[mask] *= 1.5

        for year in [2022, 2023, 2024]:
            for d in range(8, 16):
                try:
                    bbd = date(year, 10, d).toordinal()
                    t = (d - 8) / 7.0
                    mult = 2.2 + (3.0 - 2.2) * t
                    mask = daily_ordinals == bbd
                    hw[mask] *= mult
                except ValueError:
                    pass

        payday_mask = (daily_days == 1) | (daily_days >= 28)
        hw[payday_mask] *= 1.3

        sat_mask = daily_weekdays == 5
        sun_mask = daily_weekdays == 6
        hw[sat_mask] *= 1.15
        hw[sun_mask] *= 0.85

        month_weights = np.array([
            0.9, 0.85, 0.85, 0.8, 0.78, 0.8,
            0.82, 0.85, 0.95, 1.25, 1.45, 1.3
        ])
        hw *= month_weights[daily_months - 1]

        base_year = self.start_date.year
        year_factors = 1.0 + (daily_years - base_year) * 0.18
        hw *= year_factors

        rng = np.random.RandomState(self.random_state)
        hw *= rng.uniform(0.75, 1.25, size=self.total_days)

        self.date_weights = hw

    def _sample_order_dates(self, n):
        probs = self.date_weights / self.date_weights.sum()
        idx = np.random.choice(self.total_days, size=n, replace=True, p=probs)
        dates_sampled = self.daily_dates[idx]
        sort_idx = np.argsort(dates_sampled)
        return dates_sampled[sort_idx]

    def _assign_customers_vectorized(self, order_dates):
        n = len(order_dates)
        result = np.empty(n, dtype=np.int64)
        unique_dates, inv_idx = np.unique(order_dates, return_inverse=True)
        for u_idx in range(len(unique_dates)):
            od = unique_dates[u_idx]
            mask = inv_idx == u_idx
            n_for_date = mask.sum()
            eligible_mask = self.cust_signup_sorted <= od
            if not eligible_mask.any():
                eligible_mask = np.ones(len(self.cust_ids_sorted), dtype=bool)
            eligible_ids = self.cust_ids_sorted[eligible_mask]
            eligible_weights = self.cust_combined_weights[eligible_mask]
            ew_sum = eligible_weights.sum()
            if ew_sum <= 0:
                probs = None
            else:
                probs = eligible_weights / ew_sum
            chosen = np.random.choice(eligible_ids, size=n_for_date, replace=True, p=probs)
            result[mask] = chosen
        return result

    def _assign_campaigns_vectorized(self, order_dates_arr):
        n = len(order_dates_arr)
        result = np.full(n, None, dtype=object)
        for camp_idx in range(len(self.campaign_ids)):
            cid = self.campaign_ids[camp_idx]
            cs = self.campaign_starts[camp_idx]
            ce = self.campaign_ends[camp_idx]
            in_range = (order_dates_arr >= cs) & (order_dates_arr <= ce)
            if in_range.any():
                rng_c = np.random.RandomState(self.random_state + int(camp_idx))
                n_in = int(in_range.sum())
                r = rng_c.rand(n_in)
                assign_mask = in_range.copy()
                assign_mask[in_range] = r < 0.45
                empty_mask = (result == None) & assign_mask
                empty_where = np.where(empty_mask)[0]
                if len(empty_where) > 0:
                    result[empty_where] = int(cid)
        return result

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        rng_main = np.random.RandomState(self.random_state)

        n = self.num_orders

        order_dates_sorted = self._sample_order_dates(n)
        order_dates_dt = pd.to_datetime(order_dates_sorted)

        order_ids = np.arange(1, n + 1, dtype=np.int64)

        statuses = ["Delivered", "Cancelled", "Returned", "Processing", "Shipped"]
        status_probs = [0.88, 0.05, 0.04, 0.02, 0.01]
        order_statuses = rng_main.choice(statuses, size=n, p=status_probs)

        devices = ["Mobile", "Desktop", "Tablet"]
        device_probs = [0.62, 0.28, 0.10]
        device_types = rng_main.choice(devices, size=n, p=device_probs)

        customer_ids = self._assign_customers_vectorized(order_dates_sorted)

        store_ids_arr = rng_main.choice(self.store_ids, size=n, replace=True)

        campaign_ids_arr = self._assign_campaigns_vectorized(order_dates_sorted)

        num_items_per_order = rng_main.poisson(2.0, size=n)
        num_items_per_order = np.clip(num_items_per_order, 1, 8).astype(np.int64)

        total_items = int(num_items_per_order.sum())

        order_item_ids = np.arange(1, total_items + 1, dtype=np.int64)

        order_id_repeated = np.repeat(order_ids, num_items_per_order)

        product_ids_sel = rng_main.choice(self.active_products, size=total_items, replace=True)

        quantities = rng_main.poisson(1.3, size=total_items)
        quantities = np.maximum(quantities, 1).astype(np.int64)

        pid_int = product_ids_sel.astype(np.int64)
        within_lookup = pid_int <= self._max_pid
        unit_prices = np.zeros(total_items, dtype=np.float64)
        if within_lookup.any():
            unit_prices[within_lookup] = self._price_lookup[pid_int[within_lookup]]
        fallback_mask = ~within_lookup
        if fallback_mask.any():
            for fb_idx in np.where(fallback_mask)[0]:
                unit_prices[fb_idx] = self.selling_price_map.get(int(pid_int[fb_idx]), 0.0)

        camp_repeated = np.repeat(campaign_ids_arr, num_items_per_order)
        status_repeated = np.repeat(order_statuses, num_items_per_order)
        has_campaign = (camp_repeated != None) & (status_repeated != "Cancelled")

        discount_pcts = np.zeros(total_items, dtype=np.float64)

        n_has_camp_int = int(has_campaign.sum())
        if n_has_camp_int > 0:
            discount_pcts[has_campaign] = rng_main.uniform(0.05, 0.35, size=n_has_camp_int)

        no_camp_mask = ~has_campaign
        n_no_camp_int = int(no_camp_mask.sum())
        if n_no_camp_int > 0:
            r25 = rng_main.rand(n_no_camp_int) < 0.25
            n25_int = int(r25.sum())
            if n25_int > 0:
                sub_idx = np.where(no_camp_mask)[0][r25]
                discount_pcts[sub_idx] = rng_main.uniform(0.01, 0.10, size=n25_int)

        discount_amounts = np.round(quantities * unit_prices * discount_pcts, 2)
        sub_totals_before = quantities * unit_prices
        taxable = sub_totals_before - discount_amounts
        taxes = np.round(taxable * 0.18, 2)
        line_totals = np.round(taxable + taxes, 2)

        item_created_at = datetime.now()

        order_items_df = pd.DataFrame({
            "order_item_id": order_item_ids,
            "order_id": order_id_repeated,
            "product_id": product_ids_sel.astype(np.int64),
            "quantity": quantities,
            "unit_price": unit_prices,
            "discount": discount_amounts,
            "discount_pct": np.round(discount_pcts * 100, 2),
            "tax": taxes,
            "line_total": line_totals,
            "created_at": item_created_at,
        })

        agg_dict = {
            "discount": "sum",
            "tax": "sum",
            "line_total": "sum",
        }
        order_aggs = order_items_df.groupby("order_id").agg(agg_dict)
        order_aggs = order_aggs.reindex(order_ids).fillna(0.0)

        order_discount_amount = order_aggs["discount"].to_numpy()
        order_tax_amount = order_aggs["tax"].to_numpy()
        order_line_sum = order_aggs["line_total"].to_numpy()

        order_total_before_shipping = np.round(order_line_sum, 2)
        shipping_costs = np.where(
            order_total_before_shipping >= 500,
            0.0,
            rng_main.uniform(20, 80, size=n)
        )
        shipping_costs = np.round(shipping_costs, 2)
        order_totals_final = np.round(order_total_before_shipping + shipping_costs, 2)

        order_ordinals = np.array([d.toordinal() for d in order_dates_sorted], dtype=np.int64)
        end_ordinal = self.end_date.toordinal()

        shipping_ordinals = np.full(n, -1, dtype=np.int64)
        delivery_ordinals = np.full(n, -1, dtype=np.int64)

        delivered_mask = (order_statuses == "Delivered") | (order_statuses == "Returned")
        shipped_mask = order_statuses == "Shipped"

        n_deliv = int(delivered_mask.sum())
        if n_deliv > 0:
            ship_del = rng_main.randint(1, 5, size=n_deliv).astype(np.int64)
            del_del = rng_main.randint(2, 9, size=n_deliv).astype(np.int64)
            deliv_indices = np.where(delivered_mask)[0]
            sd_ord = order_ordinals[deliv_indices] + ship_del
            dd_ord = sd_ord + del_del
            over = dd_ord > end_ordinal
            if over.any():
                dd_ord[over] = end_ordinal
                sd_ord[over] = np.minimum(sd_ord[over], dd_ord[over])
            shipping_ordinals[deliv_indices] = sd_ord
            delivery_ordinals[deliv_indices] = dd_ord

        n_shipped = int(shipped_mask.sum())
        if n_shipped > 0:
            ship_sh = rng_main.randint(0, 3, size=n_shipped).astype(np.int64)
            ship_indices = np.where(shipped_mask)[0]
            shipping_ordinals[ship_indices] = order_ordinals[ship_indices] + ship_sh

        def ord_to_date_arr(ords, default_val):
            result = np.full(len(ords), default_val, dtype=object)
            valid = ords >= 0
            if valid.any():
                valid_ords_int = ords[valid].astype(np.int64)
                n_valid = len(valid_ords_int)
                valid_dates = np.empty(n_valid, dtype=object)
                for k in range(n_valid):
                    valid_dates[k] = date.fromordinal(int(valid_ords_int[k]))
                result[valid] = valid_dates
            return result

        shipping_dates = ord_to_date_arr(shipping_ordinals, None)
        delivery_dates = ord_to_date_arr(delivery_ordinals, None)

        payment_ids = np.arange(1, n + 1, dtype=np.int64)

        pay_methods_list = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery", "Wallet"]
        pay_probs = [0.35, 0.25, 0.22, 0.10, 0.06, 0.02]
        payment_methods = rng_main.choice(pay_methods_list, size=n, p=pay_probs)

        payment_statuses = np.full(n, "Success", dtype=object)

        returned_mask = order_statuses == "Returned"
        payment_statuses[returned_mask] = "Refunded"

        processing_mask = order_statuses == "Processing"
        payment_statuses[processing_mask] = "Pending"

        cancelled_mask = order_statuses == "Cancelled"
        n_canc = int(cancelled_mask.sum())
        if n_canc > 0:
            payment_statuses[cancelled_mask] = rng_main.choice(
                ["Failed", "Refunded"], size=n_canc, replace=True
            )

        hours_arr = rng_main.randint(0, 24, size=n).astype(np.int64)
        mins_arr = rng_main.randint(0, 60, size=n).astype(np.int64)
        secs_arr = rng_main.randint(0, 60, size=n).astype(np.int64)

        base_dt = pd.to_datetime(order_dates_sorted)
        order_dates_with_time_arr = (
            base_dt +
            pd.to_timedelta(hours_arr, unit='h') +
            pd.to_timedelta(mins_arr, unit='m')
        )

        transaction_dates_arr = (
            base_dt +
            pd.to_timedelta(hours_arr, unit='h') +
            pd.to_timedelta(mins_arr, unit='m') +
            pd.to_timedelta(secs_arr, unit='s')
        )

        pay_amounts = order_totals_final.copy()
        refunded_mask = payment_statuses == "Refunded"
        n_ref = int(refunded_mask.sum())
        if n_ref > 0:
            pay_amounts[refunded_mask] = np.round(
                order_totals_final[refunded_mask] * rng_main.uniform(0.5, 1.0, size=n_ref),
                2
            )

        card_last4 = np.full(n, None, dtype=object)
        card_mask = (payment_methods == "Credit Card") | (payment_methods == "Debit Card")
        n_card = int(card_mask.sum())
        if n_card > 0:
            card_numbers = rng_main.randint(1000, 10000, size=n_card).astype(str)
            card_indices = np.where(card_mask)[0]
            for k, ci in enumerate(card_indices):
                card_last4[ci] = str(card_numbers[k])

        bank_names_list = ["HDFC Bank", "SBI", "ICICI Bank", "Axis Bank", "Kotak Mahindra", "Yes Bank", "Punjab National"]
        bank_names = np.full(n, None, dtype=object)
        bank_mask = card_mask | (payment_methods == "Net Banking")
        n_bank = int(bank_mask.sum())
        if n_bank > 0:
            bank_vals = rng_main.choice(bank_names_list, size=n_bank)
            bank_indices = np.where(bank_mask)[0]
            for k, bi in enumerate(bank_indices):
                bank_names[bi] = str(bank_vals[k])

        upi_handles = ["okhdfcbank", "oksbi", "okicici", "okaxis", "paytm", "ybl", "apl"]
        upi_ids = np.full(n, None, dtype=object)
        upi_mask = payment_methods == "UPI"
        n_upi = int(upi_mask.sum())
        if n_upi > 0:
            handles = rng_main.choice(upi_handles, size=n_upi)
            upi_indices = np.where(upi_mask)[0]
            for k, ui in enumerate(upi_indices):
                upi_ids[ui] = f"{handles[k]}@upi"

        pay_created_at = datetime.now()

        payments_df = pd.DataFrame({
            "payment_id": payment_ids,
            "customer_id": customer_ids.astype(np.int64),
            "payment_method": payment_methods,
            "payment_status": payment_statuses,
            "transaction_date": transaction_dates_arr,
            "amount": pay_amounts,
            "card_last4": card_last4,
            "bank_name": bank_names,
            "upi_id": upi_ids,
            "created_at": pay_created_at,
        })

        campaigns_final_arr = np.full(n, None, dtype=object)
        not_none = campaign_ids_arr != None
        if not_none.any():
            campaigns_final_arr[not_none] = np.array([
                int(c) for c in campaign_ids_arr[not_none]
            ], dtype=object)

        now = datetime.now()

        orders_df = pd.DataFrame({
            "order_id": order_ids,
            "customer_id": customer_ids.astype(np.int64),
            "order_date": order_dates_with_time_arr,
            "order_status": order_statuses,
            "store_id": store_ids_arr.astype(np.int64),
            "payment_id": payment_ids,
            "campaign_id": campaigns_final_arr,
            "shipping_date": shipping_dates,
            "delivery_date": delivery_dates,
            "shipping_cost": shipping_costs,
            "discount_amount": np.round(order_discount_amount, 2),
            "tax_amount": np.round(order_tax_amount, 2),
            "order_total": order_totals_final,
            "device_type": device_types,
            "created_at": now,
            "updated_at": now,
        })

        return orders_df, order_items_df, payments_df
