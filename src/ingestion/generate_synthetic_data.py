import random
from datetime import datetime, timedelta, date
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from faker import Faker

from src.ingestion.constants import (
    CITIES,
    CITY_TO_STATE,
    CUSTOMER_SEGMENTS,
    PRODUCT_CATEGORIES,
    CATEGORY_WEIGHTS,
    SUPPLIER_NAMES,
    STORE_NAMES,
    PAYMENT_METHODS,
    ORDER_STATUSES,
    RETURN_REASONS,
    MARKETING_CHANNELS,
    DEVICE_TYPES,
    EMPLOYEE_ROLES,
    FIRST_NAMES_M,
    FIRST_NAMES_F,
    LAST_NAMES,
)
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


class SyntheticDataGenerator:
    def __init__(
        self,
        random_state: int = None,
        start_date: str = None,
        end_date: str = None,
        num_customers: int = None,
        num_products: int = None,
        num_orders: int = None,
    ):
        self.random_state = random_state or settings.SEED_RANDOM_STATE
        self.start_date = datetime.strptime(start_date or settings.DATA_START_DATE, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date or settings.DATA_END_DATE, "%Y-%m-%d").date()

        self.num_customers = num_customers or settings.NUM_CUSTOMERS
        self.num_products = num_products or settings.NUM_PRODUCTS
        self.num_orders = num_orders or settings.NUM_ORDERS

        self.fake = Faker("en_IN")
        Faker.seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        self._df_cache: Dict[str, pd.DataFrame] = {}

    def _cached(self, key: str, generator_fn):
        if key not in self._df_cache:
            self._df_cache[key] = generator_fn()
        return self._df_cache[key]

    # =====================================================================
    # UTILITY METHODS
    # =====================================================================

    def _weighted_choice(
        self, items_weights: List[Tuple[Any, float]], size: int = 1
    ) -> List[Any]:
        items = [i for i, _ in items_weights]
        weights = [w for _, w in items_weights]
        total = sum(weights)
        probs = [w / total for w in weights]
        if size == 1:
            return [random.choices(items, weights=probs, k=1)[0]]
        return list(random.choices(items, weights=probs, k=size))

    def _random_dates(
        self,
        start: date = None,
        end: date = None,
        size: int = 1,
    ) -> List[date]:
        start = start or self.start_date
        end = end or self.end_date
        total_days = (end - start).days
        if total_days <= 0:
            return [start] * size

        day_offsets = np.random.randint(0, total_days + 1, size=size)
        return [start + timedelta(days=int(d)) for d in day_offsets]

    def _random_date_in_year(self, year: int, size: int = 1) -> List[date]:
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        return self._random_dates(start, end, size)

    def _random_email(self, first: str, last: str) -> str:
        domains = ["gmail.com", "yahoo.in", "outlook.com", "hotmail.com", "rediffmail.com"]
        num = random.randint(0, 999)
        return f"{first.lower()}.{last.lower()}{num}@{random.choice(domains)}"

    def _random_phone(self) -> str:
        prefixes = ["9", "8", "7", "6"]
        return f"+91 {random.choice(prefixes)}{random.randint(100000000, 999999999)}"

    def _pick_from_list(self, items: List[Any], size: int = 1) -> List[Any]:
        if size == 1:
            return [random.choice(items)]
        return random.choices(items, k=size)

    # =====================================================================
    # 1. CUSTOMERS
    # =====================================================================

    def generate_customers(self) -> pd.DataFrame:
        logger.info(f"Generating {self.num_customers} customers...")

        genders = np.random.choice(["M", "F"], size=self.num_customers, p=[0.52, 0.48])
        first_names = []
        for g in genders:
            pool = FIRST_NAMES_M if g == "M" else FIRST_NAMES_F
            first_names.append(random.choice(pool))
        last_names = random.choices(LAST_NAMES, k=self.num_customers)

        cities = np.random.choice(CITIES, size=self.num_customers)
        states = [CITY_TO_STATE[c] for c in cities]

        dob_years = np.random.randint(1970, 2007, size=self.num_customers)
        dobs = [date(y, np.random.randint(1, 13), np.random.randint(1, 28)) for y in dob_years]
        ages = [self.end_date.year - d.year for d in dobs]

        signup_start = self.start_date
        signup_end = self.end_date
        signup_days_offsets = np.random.exponential(
            scale=(self.end_date - self.start_date).days / 2.5, size=self.num_customers
        ).astype(int)
        signup_days_offsets = np.clip(signup_days_offsets, 0, (self.end_date - self.start_date).days)
        signup_dates = [signup_start + timedelta(days=int(d)) for d in signup_days_offsets]

        channels = np.random.choice(
            ["Organic Search", "Direct", "Social Media", "Email", "Referral", "Paid Ads"],
            size=self.num_customers,
            p=[0.28, 0.20, 0.22, 0.10, 0.08, 0.12],
        )

        segment_probs = [w for _, w in CUSTOMER_SEGMENTS]
        segments = np.random.choice(
            [s for s, _ in CUSTOMER_SEGMENTS], size=self.num_customers, p=segment_probs
        )

        df = pd.DataFrame({
            "customer_id": range(1, self.num_customers + 1),
            "first_name": first_names,
            "last_name": last_names,
            "gender": genders,
            "date_of_birth": dobs,
            "age": ages,
            "city": cities,
            "state": states,
            "country": ["India"] * self.num_customers,
            "signup_date": signup_dates,
            "customer_segment": segments,
            "signup_channel": channels,
            "phone": [self._random_phone() for _ in range(self.num_customers)],
            "email": [self._random_email(f, l) for f, l in zip(first_names, last_names)],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        })

        self._df_cache["customers"] = df
        logger.info(f"Generated {len(df)} customers")
        return df

    # =====================================================================
    # 2. CATEGORIES
    # =====================================================================

    def generate_categories(self) -> pd.DataFrame:
        logger.info("Generating categories...")
        rows = []
        cat_id = 1
        for cat_name, subcats in PRODUCT_CATEGORIES:
            for subcat_name, _ in subcats:
                rows.append({
                    "category_id": cat_id,
                    "category_name": cat_name,
                    "subcategory": subcat_name,
                    "category_level": "L2",
                    "description": f"{subcat_name} products under {cat_name}",
                    "created_at": datetime.now(),
                })
                cat_id += 1

        df = pd.DataFrame(rows)
        self._df_cache["categories"] = df
        logger.info(f"Generated {len(df)} categories")
        return df

    # =====================================================================
    # 3. SUPPLIERS
    # =====================================================================

    def generate_suppliers(self) -> pd.DataFrame:
        logger.info("Generating suppliers...")
        num_suppliers = len(SUPPLIER_NAMES)
        rows = []
        for i in range(num_suppliers):
            city = random.choice(CITIES)
            state = CITY_TO_STATE[city]
            first = random.choice(FIRST_NAMES_M)
            last = random.choice(LAST_NAMES)
            rows.append({
                "supplier_id": i + 1,
                "supplier_name": SUPPLIER_NAMES[i],
                "contact_name": f"{first} {last}",
                "city": city,
                "state": state,
                "country": "India",
                "phone": self._random_phone(),
                "email": f"contact@{SUPPLIER_NAMES[i].lower().replace(' ', '_').replace('.', '')}.com",
                "rating": round(random.uniform(3.2, 4.9), 2),
                "lead_time_days": random.randint(3, 15),
                "reliability_score": round(random.uniform(72.0, 98.5), 2),
                "created_at": datetime.now(),
            })

        df = pd.DataFrame(rows)
        self._df_cache["suppliers"] = df
        logger.info(f"Generated {len(df)} suppliers")
        return df

    # =====================================================================
    # 4. PRODUCTS
    # =====================================================================

    def generate_products(self) -> pd.DataFrame:
        logger.info(f"Generating {self.num_products} products...")
        categories = self._cached("categories", self.generate_categories)
        suppliers = self._cached("suppliers", self.generate_suppliers)

        subcat_weight_tuples = []
        for cat_idx, (cat_name, subcats) in enumerate(PRODUCT_CATEGORIES):
            cat_w = CATEGORY_WEIGHTS[cat_idx]
            for subcat_name, sub_w in subcats:
                subcat_weight_tuples.append(((cat_name, subcat_name), cat_w * sub_w))

        product_names_by_subcat = self._build_product_names()

        rows = []
        for pid in range(1, self.num_products + 1):
            cat_name, subcat_name = self._weighted_choice(subcat_weight_tuples)[0]
            cat_row = categories[
                (categories["category_name"] == cat_name)
                & (categories["subcategory"] == subcat_name)
            ].iloc[0]

            names_pool = product_names_by_subcat.get((cat_name, subcat_name), [f"{subcat_name} Item"])
            product_name = random.choice(names_pool)
            if len(names_pool) > 1:
                model_num = f" MK{random.randint(1, 99):02d}"
                product_name = f"{product_name}{model_num}"

            supplier_id = random.choice(suppliers["supplier_id"].tolist())
            cost_price = self._generate_cost_price(cat_name, subcat_name)
            margin = self._generate_margin(cat_name, subcat_name)
            selling_price = round(cost_price * (1 + margin), 0)

            launch_start = self.start_date - timedelta(days=random.randint(30, 540))
            launch_end = self.end_date - timedelta(days=random.randint(30, 180))
            launch_date = self._random_dates(launch_start, launch_end)[0]

            status_rand = random.random()
            if status_rand < 0.90:
                product_status = "Active"
            elif status_rand < 0.97:
                product_status = "Discontinued"
            else:
                product_status = "Out of Season"

            rows.append({
                "product_id": pid,
                "product_name": product_name,
                "category_id": int(cat_row["category_id"]),
                "supplier_id": int(supplier_id),
                "sku_code": f"SKU{pid:06d}",
                "cost_price": float(cost_price),
                "selling_price": float(selling_price),
                "launch_date": launch_date,
                "weight_kg": round(random.uniform(0.05, 15.0), 2),
                "product_status": product_status,
                "brand_name": self._random_brand(cat_name),
                "description": f"Premium {subcat_name} - {product_name}",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            })

        df = pd.DataFrame(rows)
        self._df_cache["products"] = df
        logger.info(f"Generated {len(df)} products")
        return df

    def _build_product_names(self) -> Dict[Tuple[str, str], List[str]]:
        mapping = {
            ("Electronics", "Smartphones"): ["iPhone", "Samsung Galaxy", "OnePlus", "Redmi Note", "OPPO Reno", "Vivo V", "Realme GT", "Google Pixel", "Xiaomi Mi", "Nothing Phone"],
            ("Electronics", "Laptops"): ["MacBook Pro", "Dell XPS", "HP Pavilion", "Lenovo ThinkPad", "ASUS ZenBook", "Acer Aspire", "MSI Gaming", "Microsoft Surface", "MacBook Air", "ROG Strix"],
            ("Electronics", "Tablets"): ["iPad Pro", "iPad Air", "Samsung Galaxy Tab", "Lenovo Tab", "Amazon Fire HD", "Microsoft Surface Go", "OnePlus Pad", "Redmi Pad", "OPPO Pad", "Xiaomi Pad"],
            ("Electronics", "Smart Watches"): ["Apple Watch", "Samsung Galaxy Watch", "Fitbit Versa", "Garmin Venu", "Amazfit GTR", "Noise ColorFit", "boAt Watch", "Redmi Watch", "OnePlus Watch", "Fossil Gen"],
            ("Electronics", "Wireless Headphones"): ["Sony WH", "Bose QuietComfort", "Apple AirPods Pro", "Sennheiser Momentum", "JBL Tune", "boAt Rockerz", "Noise Buds", "OnePlus Buds", "Samsung Galaxy Buds", "Jabra Elite"],
            ("Electronics", "Bluetooth Speakers"): ["JBL Flip", "Sony SRS", "Bose SoundLink", "UE Boom", "Marshall Emberton", "Anker Soundcore", "Amazon Echo", "Google Nest Audio", "boAt Stone", "Portronics POR"],
            ("Electronics", "Smart TV"): ["Sony Bravia", "Samsung QLED", "LG OLED", "OnePlus TV", "Redmi Smart TV", "Mi TV", "VU Cinema", "TCL Plex", "Hisense UHD", "Panasonic TH"],
            ("Electronics", "Cameras"): ["Canon EOS", "Nikon Z", "Sony Alpha", "Fujifilm X", "Olympus OM", "Panasonic Lumix", "GoPro Hero", "DJI Osmo", "Insta360", "Ricoh GR"],
            ("Electronics", "Gaming Consoles"): ["PlayStation", "Xbox Series", "Nintendo Switch", "Steam Deck", "Valve Index", "Logitech G", "Razer Kishi", "Astro A50", "8BitDo", "Analogue Pocket"],
            ("Electronics", "Accessories"): ["USB-C Hub", "Power Bank", "Charger Adapter", "Screen Guard", "Case Cover", "Laptop Bag", "Mouse Pad", "Webcam", "USB Cable", "Portable Charger"],

            ("Fashion", "Men's Clothing"): ["Cotton Shirt", "Slim Fit Jeans", "Formal Trousers", "Polo T-Shirt", "Hooded Sweatshirt", "Denim Jacket", "Casual Blazer", "Woolen Sweater", "Track Pants", "Kurta Pyjama"],
            ("Fashion", "Women's Clothing"): ["Cotton Kurti", "Designer Saree", "Lehenga Choli", "Anarkali Suit", "Palazzo Set", "T-Shirt Dress", "Denim Jeans", "Formal Blouse", "Cardigan Sweater", "Ethnic Gown"],
            ("Fashion", "Kids' Clothing"): ["Cotton T-Shirt", "Denim Jeans", "Frock Dress", "Shorts Set", "Tracksuit", "Party Wear", "School Uniform", "Winter Jacket", "Pyjama Set", "Swimwear"],
            ("Fashion", "Footwear"): ["Running Shoes", "Casual Sneakers", "Formal Oxford", "Sports Sandals", "Leather Boots", "Flip Flops", "Loafers", "Heels", "Wedges", "Sports Shoes"],
            ("Fashion", "Watches"): ["Analog Watch", "Chronograph", "Smart Watch", "Leather Strap", "Metal Band", "Diving Watch", "Luxury Watch", "Sports Watch", "Classic Watch", "Minimalist Watch"],
            ("Fashion", "Jewelry"): ["Gold Ring", "Silver Necklace", "Diamond Earrings", "Platinum Bracelet", "Gold Mangalsutra", "Silver Pendant", "Kundan Set", "Temple Jewelry", "Beaded Necklace", "Fashion Ring"],
            ("Fashion", "Bags & Luggage"): ["Office Backpack", "Travel Trolley", "Ladies Handbag", "Laptop Bag", "Duffel Bag", "Sling Bag", "Clutch Purse", "Shoulder Bag", "Wheeled Suitcase", "Messenger Bag"],
            ("Fashion", "Sunglasses"): ["Aviator Sunglasses", "Wayfarer Shades", "Round Sunglasses", "Cat Eye", "Polarized Glasses", "Sports Goggles", "Retro Square", "Oversized Shades", "Clubmaster", "Wrap Around"],

            ("Home & Kitchen", "Kitchen Appliances"): ["Mixer Grinder", "Microwave Oven", "Refrigerator", "Washing Machine", "Air Fryer", "Electric Kettle", "Toaster", "Induction Cooktop", "Coffee Maker", "Food Processor"],
            ("Home & Kitchen", "Furniture"): ["Sofa Set", "Dining Table", "Bed with Storage", "Wardrobe", "Office Desk", "Recliner Chair", "Bookshelf", "TV Unit", "Coffee Table", "Shoe Rack"],
            ("Home & Kitchen", "Home Decor"): ["Wall Painting", "LED Lamp", "Indoor Plant", "Wall Clock", "Cushion Covers", "Carpet Rug", "Curtains", "Photo Frame", "Decorative Vase", "Wall Shelf"],
            ("Home & Kitchen", "Bedding"): ["Bedsheet Set", "Memory Pillow", "Mattress Protector", "Comforter Set", "Blanket", "Duvet Cover", "Quilt", "Mattress Topper", "Sleeping Bag", "Pillow Cover Set"],
            ("Home & Kitchen", "Cookware"): ["Non-Stick Pan", "Pressure Cooker", "Stainless Steel Kadai", "Tawa Griddle", "Casserole Set", "Tempered Glass Lid", "Aluminum Tope", "Cast Iron Skillet", "Milk Pan", "Handi Set"],
            ("Home & Kitchen", "Cleaning Supplies"): ["Liquid Detergent", "Floor Cleaner", "Dishwash Liquid", "Disinfectant Spray", "Scrub Brush", "Mop Set", "Gloves", "Sponge Wipes", "Glass Cleaner", "Toilet Cleaner"],
            ("Home & Kitchen", "Storage"): ["Plastic Containers", "Storage Boxes", "Wire Basket", "Kitchen Jar Set", "Drawer Organizer", "Spice Rack", "Laundry Basket", "Food Storage Bags", "Trash Can", "Vacuum Bags"],

            ("Beauty & Personal Care", "Skincare"): ["Face Wash", "Moisturizer Cream", "Sunscreen Lotion", "Serum", "Toner", "Face Mask", "Eye Cream", "Night Cream", "Day Cream", "Scrub Exfoliator"],
            ("Beauty & Personal Care", "Makeup"): ["Lipstick", "Foundation", "Mascara", "Kajal", "Eyeliner", "Blush", "Highlighter", "Makeup Brush Set", "Nail Polish", "Makeup Remover"],
            ("Beauty & Personal Care", "Haircare"): ["Shampoo", "Hair Conditioner", "Hair Oil", "Hair Serum", "Hair Gel", "Hair Color", "Hair Dryer", "Straightener", "Hair Mask", "Leave-In Conditioner"],
            ("Beauty & Personal Care", "Fragrances"): ["Eau de Parfum", "Body Mist", "Deodorant", "Attar Perfume", "Room Spray", "Aftershave", "Essential Oil Set", "Scented Candle", "Diffuser Oil", "Pocket Perfume"],
            ("Beauty & Personal Care", "Personal Care"): ["Toothpaste", "Toothbrush", "Soap Bar", "Body Wash", "Shaving Cream", "Razor", "Deodorant Stick", "Mouthwash", "Talc Powder", "Hand Sanitizer"],

            ("Sports & Outdoors", "Fitness Equipment"): ["Treadmill", "Dumbbell Set", "Resistance Bands", "Yoga Mat", "Exercise Cycle", "Push-Up Bar", "Kettlebell", "Pull-Up Bar", "Gym Ball", "Jump Rope"],
            ("Sports & Outdoors", "Sportswear"): ["Dry-Fit T-Shirt", "Sports Shorts", "Running Jacket", "Gym Leggings", "Sports Bra", "Track Jacket", "Compression Pants", "Swimsuit", "Cycling Jersey", "Badminton Shirt"],
            ("Sports & Outdoors", "Outdoor Gear"): ["Tent", "Sleeping Bag", "Hiking Boots", "Backpack", "Camping Stove", "Trekking Poles", "Headlamp", "Cooler Box", "Water Bottle", "First Aid Kit"],
            ("Sports & Outdoors", "Cycling"): ["Mountain Bike", "City Bicycle", "Helmet", "Cycling Gloves", "Cycle Light", "Bike Lock", "Water Bottle Cage", "Pannier Bag", "Cycle Repair Kit", "Saddle Cover"],

            ("Books & Media", "Books"): ["Fiction Novel", "Self-Help Book", "Biography", "Children's Book", "Textbook", "Cookbook", "Manga Comic", "Poetry Collection", "Business Book", "Science Book"],
            ("Books & Media", "Stationery"): ["Notebook", "Pen Set", "Pencil Box", "Diary", "Folders", "Sticky Notes", "Highlighter", "Art Supplies", "Planner", "Stapler"],
            ("Books & Media", "Toys & Games"): ["Building Blocks", "Doll Set", "Remote Control Car", "Board Game", "Puzzle", "Action Figure", "Art Set", "Musical Toy", "Soft Toy", "Educational Toy"],

            ("Automotive", "Car Accessories"): ["Car Cover", "Seat Cover", "Floor Mats", "Dashboard Perfume", "Tyre Inflator", "Car Vacuum", "Sun Shade", "Phone Mount", "Trunk Organizer", "Steering Cover"],
            ("Automotive", "Bike Accessories"): ["Bike Cover", "Helmet", "Gloves", "Bike Lock", "Mobile Holder", "LED Light", "Tyre Puncture Kit", "Saddle Bag", "Rain Cover", "Tool Kit"],
            ("Automotive", "Car Care"): ["Car Wash Shampoo", "Tyre Polish", "Dashboard Spray", "Glass Cleaner", "Wax Polish", "Microfiber Cloth", "Interior Cleaner", "Engine Degreaser", "Air Freshener", "Scratch Remover"],

            ("Grocery & Gourmet", "Snacks"): ["Potato Chips", "Biscuit Pack", "Namkeen Mix", "Chocolate Bar", "Cookies", "Dry Fruits", "Granola Bar", "Popcorn", "Peanut Butter", "Chips Variety"],
            ("Grocery & Gourmet", "Beverages"): ["Tea Powder", "Coffee Beans", "Green Tea", "Energy Drink", "Juice Pack", "Soft Drink", "Protein Shake", "Herbal Tea", "Coconut Water", "Mineral Water"],
            ("Grocery & Gourmet", "Gourmet"): ["Olive Oil", "Artisan Chocolate", "Premium Spices", "Cheese Pack", "Wine Glass Set", "Truffle Oil", "Honey Jar", "Pasta", "Sushi Kit", "Gourmet Popcorn"],
        }
        return mapping

    def _generate_cost_price(self, category: str, subcategory: str) -> float:
        ranges = {
            ("Electronics", "Smartphones"): (8000, 80000),
            ("Electronics", "Laptops"): (30000, 150000),
            ("Electronics", "Tablets"): (10000, 70000),
            ("Electronics", "Smart Watches"): (2000, 50000),
            ("Electronics", "Wireless Headphones"): (800, 25000),
            ("Electronics", "Bluetooth Speakers"): (600, 20000),
            ("Electronics", "Smart TV"): (15000, 120000),
            ("Electronics", "Cameras"): (20000, 200000),
            ("Electronics", "Gaming Consoles"): (20000, 55000),
            ("Electronics", "Accessories"): (150, 4000),
            ("Fashion", "Men's Clothing"): (250, 4000),
            ("Fashion", "Women's Clothing"): (300, 8000),
            ("Fashion", "Kids' Clothing"): (150, 2500),
            ("Fashion", "Footwear"): (300, 8000),
            ("Fashion", "Watches"): (400, 25000),
            ("Fashion", "Jewelry"): (500, 80000),
            ("Fashion", "Bags & Luggage"): (400, 12000),
            ("Fashion", "Sunglasses"): (300, 8000),
            ("Home & Kitchen", "Kitchen Appliances"): (800, 45000),
            ("Home & Kitchen", "Furniture"): (3000, 80000),
            ("Home & Kitchen", "Home Decor"): (200, 5000),
            ("Home & Kitchen", "Bedding"): (400, 8000),
            ("Home & Kitchen", "Cookware"): (250, 5000),
            ("Home & Kitchen", "Cleaning Supplies"): (100, 1500),
            ("Home & Kitchen", "Storage"): (150, 3000),
            ("Beauty & Personal Care", "Skincare"): (150, 3000),
            ("Beauty & Personal Care", "Makeup"): (200, 4000),
            ("Beauty & Personal Care", "Haircare"): (120, 2500),
            ("Beauty & Personal Care", "Fragrances"): (300, 6000),
            ("Beauty & Personal Care", "Personal Care"): (60, 800),
            ("Sports & Outdoors", "Fitness Equipment"): (500, 45000),
            ("Sports & Outdoors", "Sportswear"): (300, 4000),
            ("Sports & Outdoors", "Outdoor Gear"): (400, 15000),
            ("Sports & Outdoors", "Cycling"): (2000, 35000),
            ("Books & Media", "Books"): (150, 1500),
            ("Books & Media", "Stationery"): (50, 1000),
            ("Books & Media", "Toys & Games"): (200, 5000),
            ("Automotive", "Car Accessories"): (200, 4000),
            ("Automotive", "Bike Accessories"): (150, 3000),
            ("Automotive", "Car Care"): (80, 1500),
            ("Grocery & Gourmet", "Snacks"): (50, 800),
            ("Grocery & Gourmet", "Beverages"): (60, 1500),
            ("Grocery & Gourmet", "Gourmet"): (200, 3000),
        }
        low, high = ranges.get((category, subcategory), (200, 3000))
        return round(random.uniform(low, high), 0)

    def _generate_margin(self, category: str, subcategory: str) -> float:
        margins = {
            "Electronics": 0.12,
            "Fashion": 0.50,
            "Home & Kitchen": 0.30,
            "Beauty & Personal Care": 0.45,
            "Sports & Outdoors": 0.35,
            "Books & Media": 0.30,
            "Automotive": 0.25,
            "Grocery & Gourmet": 0.18,
        }
        base = margins.get(category, 0.30)
        return base + random.uniform(-0.08, 0.12)

    def _random_brand(self, category: str) -> str:
        brands = {
            "Electronics": ["Apple", "Samsung", "Sony", "OnePlus", "Xiaomi", "boAt", "Noise", "JBL", "LG", "Dell"],
            "Fashion": ["Nike", "Adidas", "H&M", "Zara", "Levi's", "Puma", "Bata", "Allen Solly", "Peter England", "Fabindia"],
            "Home & Kitchen": ["Prestige", "Philips", "Havells", "Bajaj", "LG", "Whirlpool", "Godrej", "Butterfly", "Crompton", "USHA"],
            "Beauty & Personal Care": ["Lakme", "L'Oréal", "Himalaya", "The Man Company", "Biotique", "Maybelline", "VLCC", "Mamaearth", "Forest Essentials", "WOW Skin"],
            "Sports & Outdoors": ["Nike", "Adidas", "Puma", "Decathlon", "Sparx", "Campus", "Quechua", "Forclaz", "DOMYOS", "BTWIN"],
            "Books & Media": ["Flipkart", "Amazon", "Crossword", "Archies", "Classmate", "Navneet", "Unicorn", "BIC", "Hauser", "Reynolds"],
            "Automotive": ["Bosch", "3M", "Michelin", "Turtle Wax", "Meguiar's", "Hella", "Lumax", "Uno Minda", "Minda", "Seal"],
            "Grocery & Gourmet": ["Tata", "Haldiram's", "Parle", "Britannia", "Nestlé", "ITC", "Amul", "Patanjali", "MDH", "Everest"],
        }
        return random.choice(brands.get(category, ["Generic Brand"]))

    # =====================================================================
    # 5. STORES
    # =====================================================================

    def generate_stores(self) -> pd.DataFrame:
        logger.info("Generating stores...")
        store_cities = {
            "Mumbai Flagship": ("Mumbai", "Maharashtra"),
            "Delhi Superstore": ("New Delhi", "Delhi"),
            "Bengaluru Central": ("Bengaluru", "Karnataka"),
            "Chennai Plaza": ("Chennai", "Tamil Nadu"),
            "Hyderabad Hub": ("Hyderabad", "Telangana"),
            "Pune Retail": ("Pune", "Maharashtra"),
            "Ahmedabad Mall": ("Ahmedabad", "Gujarat"),
            "Surat Centre": ("Surat", "Gujarat"),
            "Jaipur Junction": ("Jaipur", "Rajasthan"),
            "Lucknow Point": ("Lucknow", "Uttar Pradesh"),
        }
        types = ["Warehouse", "Retail", "Fulfillment Center"]
        rows = []
        for i, name in enumerate(STORE_NAMES):
            city, state = store_cities.get(name, (random.choice(CITIES), ""))
            state = state or CITY_TO_STATE.get(city, "Maharashtra")
            rows.append({
                "store_id": i + 1,
                "store_name": name,
                "store_type": random.choice(types),
                "city": city,
                "state": state,
                "country": "India",
                "opening_date": self._random_dates(self.start_date - timedelta(days=1000), self.start_date)[0],
                "store_area_sqft": random.randint(2000, 25000),
                "store_manager_id": None,
                "created_at": datetime.now(),
            })
        df = pd.DataFrame(rows)
        self._df_cache["stores"] = df
        logger.info(f"Generated {len(df)} stores")
        return df

    # =====================================================================
    # 6. EMPLOYEES
    # =====================================================================

    def generate_employees(self) -> pd.DataFrame:
        logger.info("Generating employees...")
        stores = self._cached("stores", self.generate_stores)
        num_employees = 80
        rows = []
        for eid in range(1, num_employees + 1):
            gender = random.choice(["M", "F"])
            first = random.choice(FIRST_NAMES_M if gender == "M" else FIRST_NAMES_F)
            last = random.choice(LAST_NAMES)
            role = random.choice(EMPLOYEE_ROLES)
            store_id = random.choice(stores["store_id"].tolist())
            hire_start = self.start_date - timedelta(days=random.randint(365, 1800))
            hire_end = self.end_date
            hire_date = self._random_dates(hire_start, hire_end)[0]
            salary_map = {
                "Store Manager": (60000, 150000),
                "Operations Manager": (55000, 120000),
                "Warehouse Manager": (45000, 85000),
                "Finance Analyst": (40000, 95000),
                "Marketing Specialist": (35000, 75000),
                "Customer Service": (20000, 40000),
                "Sales Associate": (18000, 35000),
                "Inventory Clerk": (18000, 32000),
            }
            low, high = salary_map.get(role, (20000, 50000))
            salary = round(random.uniform(low, high), 0)

            dept_map = {
                "Store Manager": "Operations",
                "Operations Manager": "Operations",
                "Warehouse Manager": "Warehouse",
                "Finance Analyst": "Finance",
                "Marketing Specialist": "Marketing",
                "Customer Service": "Customer Care",
                "Sales Associate": "Sales",
                "Inventory Clerk": "Warehouse",
            }

            rows.append({
                "employee_id": eid,
                "first_name": first,
                "last_name": last,
                "gender": gender,
                "role": role,
                "department": dept_map.get(role, "Operations"),
                "store_id": store_id,
                "hire_date": hire_date,
                "salary": salary,
                "email": f"{first.lower()}.{last.lower()}{random.randint(10,99)}@company.com",
                "phone": self._random_phone(),
                "performance_score": round(random.uniform(55.0, 98.0), 2),
                "created_at": datetime.now(),
            })

        df = pd.DataFrame(rows)

        managers = df[df["role"] == "Store Manager"]["employee_id"].tolist()
        if managers and len(managers) >= len(stores):
            for idx, sid in enumerate(stores["store_id"].tolist()):
                if idx < len(managers):
                    df.loc[df["store_id"] == sid, "store_manager_id"] = managers[idx]

        self._df_cache["employees"] = df
        logger.info(f"Generated {len(df)} employees")
        return df

    # =====================================================================
    # 7. INVENTORY
    # =====================================================================

    def generate_inventory(self) -> pd.DataFrame:
        logger.info("Generating inventory records...")
        products = self._cached("products", self.generate_products)
        stores = self._cached("stores", self.generate_stores)

        rows = []
        inv_id = 1
        for pid in products["product_id"].tolist():
            active = products.loc[products["product_id"] == pid, "product_status"].iloc[0] == "Active"
            for sid in stores["store_id"].tolist():
                if not active and random.random() > 0.3:
                    continue
                base_stock = random.randint(0, 800)
                reorder = max(20, int(base_stock * 0.15))
                safety = max(10, int(base_stock * 0.08))
                daily_demand = round(random.uniform(0.5, 80.0), 2)
                rows.append({
                    "inventory_id": inv_id,
                    "product_id": int(pid),
                    "store_id": int(sid),
                    "stock_quantity": int(base_stock),
                    "reorder_level": reorder,
                    "safety_stock": safety,
                    "last_restock_date": self._random_dates(
                        self.end_date - timedelta(days=random.randint(1, 180)),
                        self.end_date
                    )[0],
                    "average_daily_demand": daily_demand,
                    "lead_time_days": random.randint(3, 14),
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                })
                inv_id += 1

        df = pd.DataFrame(rows)
        self._df_cache["inventory"] = df
        logger.info(f"Generated {len(df)} inventory records")
        return df

    # =====================================================================
    # 8. MARKETING CAMPAIGNS
    # =====================================================================

    def generate_marketing_campaigns(self) -> pd.DataFrame:
        logger.info("Generating marketing campaigns...")

        campaign_names = [
            "Big Billion Days",
            "Diwali Mega Sale",
            "New Year Offer",
            "End of Season Sale",
            "Independence Day Special",
            "Republic Day Bonanza",
            "Summer Clearance",
            "Monsoon Fest",
            "Holi Dhamaka",
            "Rakhi Special",
            "Black Friday Deals",
            "Christmas Sale",
            "Women's Day Offers",
            "Back to School",
            "Navratri Collection",
            "Eid Mubarak Sale",
            "Onam Festival",
            "Pongal Celebration",
            "Weekend Flash",
            "Midnight Madness",
            "Mega Value Deals",
            "Premium Showcase",
            "Fresh Arrivals",
            "Clearance Blowout",
            "Anniversary Fiesta",
        ]

        campaign_types = [
            "Festival Sale", "Flash Sale", "Clearance", "New Launch",
            "Loyalty Offer", "Seasonal", "Bank Offer", "Bundle Offer",
        ]

        rows = []
        for i in range(len(campaign_names)):
            year = random.choice([self.start_date.year, self.start_date.year + 1, self.end_date.year])
            start_date = self._random_date_in_year(year, 1)[0]
            duration = random.randint(3, 21)
            end_date = start_date + timedelta(days=duration)
            if end_date.year > self.end_date.year:
                end_date = self.end_date

            status_rand = random.random()
            if end_date < self.end_date - timedelta(days=14):
                status = "Completed"
            elif start_date > self.end_date:
                status = "Planned"
            else:
                status = random.choice(["Active", "Completed"])

            budget = round(random.uniform(500000, 20000000), 0)
            channel = self._weighted_choice(MARKETING_CHANNELS)[0]

            rows.append({
                "campaign_id": i + 1,
                "campaign_name": campaign_names[i % len(campaign_names)] + f" {year}",
                "campaign_type": random.choice(campaign_types),
                "channel": channel,
                "start_date": start_date,
                "end_date": end_date,
                "target_audience": random.choice(["All Customers", "High Value", "New Users", "At Risk", "Return Customers"]),
                "total_budget": budget,
                "target_revenue": round(budget * random.uniform(3.5, 8.0), 0),
                "status": status,
                "description": f"{campaign_names[i % len(campaign_names)]} with exclusive discounts across {random.choice(['all categories', 'select brands', 'premium products'])}",
                "created_at": datetime.now(),
            })

        df = pd.DataFrame(rows)
        self._df_cache["marketing_campaigns"] = df
        logger.info(f"Generated {len(df)} campaigns")
        return df

    # =====================================================================
    # 9. ORDERS + ORDER ITEMS + PAYMENTS
    # =====================================================================

    def generate_orders_and_related(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info(f"Generating {self.num_orders} orders + items + payments [VECTORIZED]...")
        from src.ingestion.generate_synthetic_data_fast import FastOrderGenerator

        customers = self._cached("customers", self.generate_customers)
        products = self._cached("products", self.generate_products)
        stores = self._cached("stores", self.generate_stores)
        campaigns = self._cached("marketing_campaigns", self.generate_marketing_campaigns)

        # Ensure campaign dates are date objects (not timestamps) for fast generator
        campaigns_fast = campaigns.copy()
        for col in ("start_date", "end_date"):
            if col in campaigns_fast.columns:
                campaigns_fast[col] = pd.to_datetime(campaigns_fast[col]).dt.date

        customers_fast = customers.copy()
        if "signup_date" in customers_fast.columns:
            customers_fast["signup_date"] = pd.to_datetime(customers_fast["signup_date"]).dt.date

        generator = FastOrderGenerator(
            customers_df=customers_fast,
            products_df=products,
            stores_df=stores,
            campaigns_df=campaigns_fast,
            start_date=self.start_date,
            end_date=self.end_date,
            num_orders=self.num_orders,
            random_state=self.random_state,
        )
        orders_df, items_df, payments_df = generator.generate()

        # Restore datetime types (fast generator returns object/date) for downstream compatibility
        for col in ("order_date", "shipping_date", "delivery_date"):
            if col in orders_df.columns:
                orders_df[col] = pd.to_datetime(orders_df[col], errors="coerce")
        for col in ("transaction_date",):
            if col in payments_df.columns:
                payments_df[col] = pd.to_datetime(payments_df[col], errors="coerce")

        self._df_cache["orders"] = orders_df
        self._df_cache["order_items"] = items_df
        self._df_cache["payments"] = payments_df

        logger.info(
            f"Generated {len(orders_df)} orders, "
            f"{len(items_df)} order items, {len(payments_df)} payments"
        )
        return orders_df, items_df, payments_df

    def _generate_order_dates(self, size: int) -> List[date]:
        total_days = (self.end_date - self.start_date).days + 1
        daily_dates = [self.start_date + timedelta(days=i) for i in range(total_days)]

        # Seasonality: Diwali/Nov highest, month-end slightly higher, weekdays > weekend
        def date_weight(d: date) -> float:
            w = 1.0
            # Month weight: 10=Oct, 11=Nov, 12=Dec festival season
            m = d.month
            month_weights = {
                1: 0.9, 2: 0.85, 3: 0.85, 4: 0.8, 5: 0.78, 6: 0.8,
                7: 0.82, 8: 0.85, 9: 0.95, 10: 1.25, 11: 1.45, 12: 1.3,
            }
            w *= month_weights.get(m, 1.0)
            # Weekend slightly lower
            if d.weekday() >= 5:
                w *= 0.88
            # Year growth
            year_factor = 1.0 + (d.year - self.start_date.year) * 0.18
            w *= year_factor
            # Randomness
            w *= random.uniform(0.75, 1.25)
            return w

        weights = [date_weight(d) for d in daily_dates]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        chosen_idx = np.random.choice(len(daily_dates), size=size, replace=True, p=probs)
        return [daily_dates[i] for i in chosen_idx]

    # =====================================================================
    # 10. RETURNS
    # =====================================================================

    def generate_returns(self) -> pd.DataFrame:
        logger.info("Generating returns...")
        orders = self._df_cache.get("orders")
        items = self._df_cache.get("order_items")
        if orders is None or items is None:
            self.generate_orders_and_related()
            orders = self._df_cache["orders"]
            items = self._df_cache["order_items"]

        returned_orders = orders[orders["order_status"] == "Returned"]["order_id"].tolist()
        # Additional partial returns from Delivered orders
        delivered_orders = orders[orders["order_status"] == "Delivered"]["order_id"].tolist()
        extra_return_orders = list(np.random.choice(
            delivered_orders, size=int(len(delivered_orders) * 0.025), replace=False
        ))
        all_return_order_ids = list(set(returned_orders) | set(extra_return_orders))

        rows = []
        rid = 1
        for oid in all_return_order_ids:
            order_row = orders[orders["order_id"] == oid].iloc[0]
            order_items = items[items["order_id"] == oid]

            return_pct = 1.0 if oid in returned_orders else random.uniform(0.1, 0.5)
            num_items_return = max(1, int(round(len(order_items) * return_pct)))
            selected = order_items.sample(n=min(num_items_return, len(order_items)), replace=False)

            for _, oi in selected.iterrows():
                base_date = order_row["delivery_date"] if pd.notna(order_row["delivery_date"]) else order_row["order_date"]
                if isinstance(base_date, datetime):
                    base_date = base_date.date()
                return_date = base_date + timedelta(days=random.randint(1, 14))
                if return_date > self.end_date:
                    return_date = self.end_date

                refund_amt = float(oi["line_total"]) * random.uniform(0.85, 1.0)

                rows.append({
                    "return_id": rid,
                    "order_item_id": int(oi["order_item_id"]),
                    "order_id": int(oid),
                    "customer_id": int(order_row["customer_id"]),
                    "product_id": int(oi["product_id"]),
                    "return_date": return_date,
                    "return_reason": random.choice(RETURN_REASONS),
                    "return_status": "Processed" if return_date < self.end_date - timedelta(days=3) else random.choice(["Approved", "Requested"]),
                    "refund_amount": round(refund_amt, 2),
                    "quantity_returned": int(oi["quantity"]),
                    "processing_days": random.randint(1, 9),
                    "created_at": datetime.now(),
                })
                rid += 1

        df = pd.DataFrame(rows)
        self._df_cache["returns"] = df
        logger.info(f"Generated {len(df)} returns")
        return df

    # =====================================================================
    # 11. REVIEWS
    # =====================================================================

    def generate_reviews(self) -> pd.DataFrame:
        logger.info("Generating reviews...")
        orders = self._df_cache.get("orders")
        items = self._df_cache.get("order_items")
        customers = self._cached("customers", self.generate_customers)
        if orders is None or items is None:
            self.generate_orders_and_related()
            orders = self._df_cache["orders"]
            items = self._df_cache["order_items"]

        delivered_or_returned = orders[
            orders["order_status"].isin(["Delivered", "Returned"])
        ]["order_id"].tolist()

        review_items = items[items["order_id"].isin(delivered_or_returned)]
        # ~35% of items get a review
        sample_size = int(len(review_items) * 0.35)
        review_sample = review_items.sample(n=sample_size, random_state=self.random_state)

        review_titles_positive = ["Excellent product!", "Worth the price", "Great quality", "Love it!", "Fast delivery", "As described", "Highly recommended", "Awesome purchase"]
        review_titles_neutral = ["Okay product", "Average quality", "Decent for the price", "Not bad", "Could be better"]
        review_titles_negative = ["Disappointed", "Poor quality", "Not worth it", "Defective item", "Wrong product", "Bad experience"]

        rows = []
        rvid = 1
        for _, ri in review_sample.iterrows():
            oid = int(ri["order_id"])
            order_row = orders[orders["order_id"] == oid].iloc[0]
            deliv_date = order_row["delivery_date"] if pd.notna(order_row["delivery_date"]) else order_row["order_date"]
            if isinstance(deliv_date, datetime):
                deliv_date = deliv_date.date()
            review_date = deliv_date + timedelta(days=random.randint(1, 30))
            if review_date > self.end_date:
                review_date = self.end_date

            # Rating distribution: realistic
            rating_rand = random.random()
            if rating_rand < 0.55:
                rating = 5
            elif rating_rand < 0.80:
                rating = 4
            elif rating_rand < 0.90:
                rating = 3
            elif rating_rand < 0.96:
                rating = 2
            else:
                rating = 1

            if rating >= 4:
                title = random.choice(review_titles_positive)
            elif rating == 3:
                title = random.choice(review_titles_neutral)
            else:
                title = random.choice(review_titles_negative)

            rows.append({
                "review_id": rvid,
                "customer_id": int(order_row["customer_id"]),
                "product_id": int(ri["product_id"]),
                "order_id": oid,
                "review_date": review_date,
                "rating": rating,
                "review_title": title,
                "review_text": f"{title} Product quality is {'good' if rating >= 4 else 'poor' if rating <= 2 else 'average'} for the price paid.",
                "helpful_votes": max(0, int(np.random.poisson(0.4))),
                "verified_purchase": True,
                "created_at": datetime.now(),
            })
            rvid += 1

        df = pd.DataFrame(rows)
        self._df_cache["reviews"] = df
        logger.info(f"Generated {len(df)} reviews")
        return df

    # =====================================================================
    # 12. MARKETING SPEND
    # =====================================================================

    def generate_marketing_spend(self) -> pd.DataFrame:
        logger.info("Generating marketing spend data...")
        campaigns = self._cached("marketing_campaigns", self.generate_marketing_campaigns)

        rows = []
        sid = 1
        for _, camp in campaigns.iterrows():
            cid = int(camp["campaign_id"])
            channel = camp["channel"]
            start = camp["start_date"]
            end = camp["end_date"]
            budget = float(camp["total_budget"])

            days = (end - start).days + 1
            # Daily weight (some days bigger launches)
            daily_weights = np.array([random.uniform(0.6, 1.6) for _ in range(days)])
            # Weekend higher or lower
            for i in range(days):
                d = start + timedelta(days=i)
                if d.weekday() >= 5:
                    daily_weights[i] *= 1.2
            daily_weights = daily_weights / daily_weights.sum()
            daily_spend_total = budget * daily_weights

            # For each day, have the main channel + 1-2 secondary
            for i in range(days):
                d = start + timedelta(days=i)
                main_spend = float(daily_spend_total[i])

                # Main channel
                impressions = int(main_spend * random.uniform(20, 80))
                ctr = random.uniform(0.01, 0.05)
                clicks = int(impressions * ctr)
                cpc = main_spend / clicks if clicks > 0 else 0
                rows.append({
                    "spend_id": sid,
                    "campaign_id": cid,
                    "spend_date": d,
                    "channel": channel,
                    "impressions": impressions,
                    "clicks": clicks,
                    "spend_amount": round(main_spend, 2),
                    "ctr": round(ctr, 6),
                    "cpc": round(cpc, 4),
                    "created_at": datetime.now(),
                })
                sid += 1

                # Secondary channels (smaller portions)
                for sec_channel, _ in random.sample(MARKETING_CHANNELS, k=random.randint(0, 2)):
                    if sec_channel == channel:
                        continue
                    sec_spend = main_spend * random.uniform(0.02, 0.15)
                    sec_impr = int(sec_spend * random.uniform(20, 80))
                    sec_ctr = random.uniform(0.005, 0.04)
                    sec_clicks = int(sec_impr * sec_ctr)
                    sec_cpc = sec_spend / sec_clicks if sec_clicks > 0 else 0
                    rows.append({
                        "spend_id": sid,
                        "campaign_id": cid,
                        "spend_date": d,
                        "channel": sec_channel,
                        "impressions": sec_impr,
                        "clicks": sec_clicks,
                        "spend_amount": round(sec_spend, 2),
                        "ctr": round(sec_ctr, 6),
                        "cpc": round(sec_cpc, 4),
                        "created_at": datetime.now(),
                    })
                    sid += 1

        df = pd.DataFrame(rows)
        self._df_cache["marketing_spend"] = df
        logger.info(f"Generated {len(df)} marketing spend records")
        return df

    # =====================================================================
    # 13. WEBSITE SESSIONS
    # =====================================================================

    def generate_website_sessions(self) -> pd.DataFrame:
        logger.info("Generating website sessions...")
        customers = self._cached("customers", self.generate_customers)
        campaigns = self._cached("marketing_campaigns", self.generate_marketing_campaigns)
        orders = self._df_cache.get("orders")

        campaign_date_map = dict(zip(campaigns["campaign_id"], zip(campaigns["start_date"], campaigns["end_date"])))
        campaign_channel_map = dict(zip(campaigns["campaign_id"], campaigns["channel"]))

        total_days = (self.end_date - self.start_date).days + 1
        # ~3x sessions as orders (industry average)
        num_sessions = self.num_orders * 3

        sessions_per_day_weight = np.zeros(total_days)
        for i in range(total_days):
            d = self.start_date + timedelta(days=i)
            m = d.month
            mw = {10: 1.4, 11: 1.6, 12: 1.3, 1: 0.95, 2: 0.9, 3: 0.9, 4: 0.85, 5: 0.8, 6: 0.8, 7: 0.85, 8: 0.9, 9: 1.0}[m]
            yf = 1.0 + (d.year - self.start_date.year) * 0.18
            we = 0.9 if d.weekday() >= 5 else 1.0
            sessions_per_day_weight[i] = mw * yf * we * random.uniform(0.8, 1.25)

        sessions_per_day_weight = sessions_per_day_weight / sessions_per_day_weight.sum()
        daily_counts = np.round(sessions_per_day_weight * num_sessions).astype(int)

        rows = []
        session_id = 1

        # Also get order dates for some sessions that convert
        order_dates_list = []
        if orders is not None:
            for od in orders["order_date"].tolist():
                if isinstance(od, datetime):
                    order_dates_list.append(od.date())

        # Pre-compute eligible customers per day for efficiency
        eligible_customers_by_day = {}
        for day_idx in range(total_days):
            d = self.start_date + timedelta(days=day_idx)
            eligible_customers_by_day[day_idx] = customers[customers["signup_date"] <= d]["customer_id"].values

        for day_idx in range(total_days):
            d = self.start_date + timedelta(days=day_idx)
            count = int(daily_counts[day_idx])
            eligible = eligible_customers_by_day[day_idx]

            for _ in range(count):
                # Customer: 60% logged in (known)
                cust_id = None
                if random.random() < 0.60 and len(eligible) > 0:
                    cust_id = int(np.random.choice(eligible))

                session_start = datetime.combine(d, datetime.min.time()) + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59),
                )

                # Channel: depends on whether there's an active campaign
                active_campaigns = []
                for cid, (s, e) in campaign_date_map.items():
                    if s <= d <= e:
                        active_campaigns.append(cid)

                campaign_id = None
                if active_campaigns and random.random() < 0.55:
                    campaign_id = random.choice(active_campaigns)
                    channel = campaign_channel_map.get(campaign_id, "Google Ads")
                else:
                    channel = self._weighted_choice(MARKETING_CHANNELS)[0]

                device = self._weighted_choice(DEVICE_TYPES)[0]
                page_views = max(1, int(np.random.poisson(8)))
                product_views = max(0, page_views - random.randint(1, 4))
                cart_adds = max(0, int(np.random.poisson(0.8))) if random.random() < 0.35 else 0
                checkout_started = 1 if cart_adds > 0 and random.random() < 0.55 else 0
                checkout_completed = 1 if checkout_started and random.random() < 0.68 else 0

                session_seconds = int(
                    page_views * random.uniform(20, 80) +
                    product_views * random.uniform(30, 90)
                )
                session_end = session_start + timedelta(seconds=session_seconds)
                bounce_rate = page_views <= 1

                rows.append({
                    "session_id": session_id,
                    "customer_id": cust_id,
                    "session_date": d,
                    "session_start": session_start,
                    "session_end": session_end,
                    "device_type": device,
                    "channel": channel,
                    "campaign_id": campaign_id,
                    "page_views": page_views,
                    "product_views": product_views,
                    "cart_adds": cart_adds,
                    "checkout_started": checkout_started,
                    "checkout_completed": checkout_completed,
                    "session_duration_sec": session_seconds,
                    "bounce_rate": bounce_rate,
                    "created_at": datetime.now(),
                })
                session_id += 1

            if day_idx % 100 == 0:
                logger.info(f"Sessions: processed day {day_idx}/{total_days}, generated {session_id-1} sessions")

        df = pd.DataFrame(rows)
        self._df_cache["website_sessions"] = df
        logger.info(f"Generated {len(df)} website sessions")
        return df

    # =====================================================================
    # FULL DATASET GENERATION
    # =====================================================================

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        logger.info("=" * 60)
        logger.info("STARTING FULL SYNTHETIC DATA GENERATION")
        logger.info(f"Random State: {self.random_state}")
        logger.info(f"Date Range: {self.start_date} to {self.end_date}")
        logger.info(f"Target Customers: {self.num_customers}")
        logger.info(f"Target Products: {self.num_products}")
        logger.info(f"Target Orders: {self.num_orders}")
        logger.info("=" * 60)

        self.generate_customers()
        self.generate_categories()
        self.generate_suppliers()
        self.generate_products()
        self.generate_stores()
        self.generate_employees()
        self.generate_inventory()
        self.generate_marketing_campaigns()
        self.generate_orders_and_related()
        self.generate_returns()
        self.generate_reviews()
        self.generate_marketing_spend()
        self.generate_website_sessions()

        logger.info("=" * 60)
        logger.info("SYNTHETIC DATA GENERATION COMPLETE")
        total_rows = sum(len(v) for v in self._df_cache.values())
        logger.info(f"Total rows across all tables: {total_rows:,}")
        for name, df in self._df_cache.items():
            logger.info(f"  {name}: {len(df):,} rows")
        logger.info("=" * 60)

        return self._df_cache

    def save_to_csv(self, output_dir: Path = None) -> None:
        output_dir = Path(output_dir) if output_dir else settings.RAW_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self._df_cache:
            self.generate_all()

        logger.info(f"Saving all datasets to {output_dir}...")
        for name, df in self._df_cache.items():
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False, encoding="utf-8")
            logger.info(f"  Saved {name}.csv ({len(df):,} rows)")
        logger.info("All datasets saved successfully")


def main():
    generator = SyntheticDataGenerator()
    datasets = generator.generate_all()
    generator.save_to_csv()
    return datasets


if __name__ == "__main__":
    main()
