INDIAN_STATES = [
    (
        "Uttar Pradesh",
        [
            "Lucknow", "Kanpur", "Varanasi", "Agra", "Prayagraj",
            "Ghaziabad", "Noida", "Meerut", "Bareilly", "Aligarh",
        ],
    ),
    (
        "Maharashtra",
        [
            "Mumbai", "Pune", "Nagpur", "Thane", "Nashik",
            "Aurangabad", "Solapur", "Amravati", "Kolhapur", "Sangli",
        ],
    ),
    (
        "Karnataka",
        [
            "Bengaluru", "Hubli", "Mysuru", "Mangaluru", "Belagavi",
            "Gulbarga", "Davanagere", "Bellary", "Bijapur", "Shimoga",
        ],
    ),
    (
        "Tamil Nadu",
        [
            "Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem",
            "Tirunelveli", "Tiruppur", "Erode", "Vellore", "Thoothukkudi",
        ],
    ),
    (
        "Gujarat",
        [
            "Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar",
            "Jamnagar", "Junagadh", "Gandhinagar", "Anand", "Navsari",
        ],
    ),
    (
        "Rajasthan",
        [
            "Jaipur", "Jodhpur", "Udaipur", "Kota", "Bikaner",
            "Ajmer", "Bhilwara", "Alwar", "Bharatpur", "Sikar",
        ],
    ),
    (
        "West Bengal",
        [
            "Kolkata", "Howrah", "Durgapur", "Asansol", "Siliguri",
            "Malda", "Bardhaman", "Bankura", "Midnapore", "Kharagpur",
        ],
    ),
    (
        "Punjab",
        [
            "Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda",
            "Mohali", "Batala", "Pathankot", "Firozpur", "Moga",
        ],
    ),
    (
        "Haryana",
        [
            "Gurugram", "Faridabad", "Panipat", "Ambala", "Yamunanagar",
            "Rohtak", "Hisar", "Karnal", "Sonipat", "Panchkula",
        ],
    ),
    (
        "Delhi",
        [
            "New Delhi", "North Delhi", "South Delhi",
            "East Delhi", "West Delhi", "Central Delhi",
        ],
    ),
    (
        "Kerala",
        [
            "Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Kollam",
            "Palakkad", "Malappuram", "Alappuzha", "Kannur", "Idukki",
        ],
    ),
    (
        "Telangana",
        [
            "Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam",
            "Ramagundam", "Mahbubnagar", "Nalgonda", "Adilabad", "Siddipet",
        ],
    ),
    (
        "Andhra Pradesh",
        [
            "Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Kurnool",
            "Rajahmundry", "Kakinada", "Tirupati", "Anantapur", "Eluru",
        ],
    ),
    (
        "Madhya Pradesh",
        [
            "Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain",
            "Sagar", "Dewas", "Satna", "Ratlam", "Rewa",
        ],
    ),
    (
        "Chhattisgarh",
        [
            "Raipur", "Bhilai", "Bilaspur", "Korba", "Durg",
            "Raigarh", "Jagdalpur", "Ambikapur", "Rajnandgaon",
        ],
    ),
    (
        "Odisha",
        [
            "Bhubaneswar", "Cuttack", "Rourkela", "Berhampur", "Sambalpur",
            "Puri", "Bhadrak", "Balasore", "Barbil", "Jharsuguda",
        ],
    ),
    (
        "Bihar",
        [
            "Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga",
            "Purnia", "Begusarai", "Katihar", "Munger", "Chhapra",
        ],
    ),
    (
        "Jharkhand",
        [
            "Ranchi", "Jamshedpur", "Dhanbad", "Bokaro", "Deoghar",
            "Phusro", "Hazaribagh", "Giridih", "Ramgarh", "Medininagar",
        ],
    ),
    (
        "Assam",
        [
            "Guwahati", "Silchar", "Dibrugarh", "Jorhat", "Nagaon",
            "Tinsukia", "Bongaigaon", "Tezpur", "Kokrajhar", "Karimganj",
        ],
    ),
    (
        "Uttarakhand",
        [
            "Dehradun", "Haridwar", "Roorkee", "Haldwani", "Rudrapur",
            "Kashipur", "Rishikesh", "Nainital", "Kotdwar", "Ramnagar",
        ],
    ),
    (
        "Goa",
        [
            "Panaji", "Margao", "Vasco da Gama", "Mapusa", "Ponda",
            "Bicholim", "Curchorem", "Sanquelim", "Cuncolim", "Quepem",
        ],
    ),
    (
        "Himachal Pradesh",
        [
            "Shimla", "Dharamshala", "Solan", "Mandi", "Palampur",
            "Kullu", "Hamirpur", "Kangra", "Una", "Bilaspur",
        ],
    ),
]

CITY_TO_STATE = {}
for state, cities in INDIAN_STATES:
    for city in cities:
        CITY_TO_STATE[city] = state

CITIES = list(CITY_TO_STATE.keys())

CUSTOMER_SEGMENTS = [
    ("Champion", 0.10),
    ("Loyal Customer", 0.15),
    ("Potential Loyalist", 0.20),
    ("New Customer", 0.20),
    ("At Risk", 0.15),
    ("Can't Lose Them", 0.10),
    ("Lost Customer", 0.10),
]

PRODUCT_CATEGORIES = [
    (
        "Electronics",
        [
            ("Smartphones", 0.18),
            ("Laptops", 0.12),
            ("Tablets", 0.06),
            ("Smart Watches", 0.08),
            ("Wireless Headphones", 0.10),
            ("Bluetooth Speakers", 0.05),
            ("Smart TV", 0.07),
            ("Cameras", 0.04),
            ("Gaming Consoles", 0.05),
            ("Accessories", 0.08),
        ],
    ),
    (
        "Fashion",
        [
            ("Men's Clothing", 0.12),
            ("Women's Clothing", 0.15),
            ("Kids' Clothing", 0.08),
            ("Footwear", 0.10),
            ("Watches", 0.06),
            ("Jewelry", 0.05),
            ("Bags & Luggage", 0.05),
            ("Sunglasses", 0.03),
        ],
    ),
    (
        "Home & Kitchen",
        [
            ("Kitchen Appliances", 0.08),
            ("Furniture", 0.10),
            ("Home Decor", 0.06),
            ("Bedding", 0.05),
            ("Cookware", 0.05),
            ("Cleaning Supplies", 0.04),
            ("Storage", 0.04),
        ],
    ),
    (
        "Beauty & Personal Care",
        [
            ("Skincare", 0.10),
            ("Makeup", 0.08),
            ("Haircare", 0.07),
            ("Fragrances", 0.05),
            ("Personal Care", 0.05),
        ],
    ),
    (
        "Sports & Outdoors",
        [
            ("Fitness Equipment", 0.07),
            ("Sportswear", 0.09),
            ("Outdoor Gear", 0.06),
            ("Cycling", 0.04),
        ],
    ),
    (
        "Books & Media",
        [
            ("Books", 0.08),
            ("Stationery", 0.05),
            ("Toys & Games", 0.07),
        ],
    ),
    (
        "Automotive",
        [
            ("Car Accessories", 0.06),
            ("Bike Accessories", 0.04),
            ("Car Care", 0.03),
        ],
    ),
    (
        "Grocery & Gourmet",
        [
            ("Snacks", 0.08),
            ("Beverages", 0.07),
            ("Gourmet", 0.03),
        ],
    ),
]

CATEGORY_WEIGHTS = [0.32, 0.28, 0.14, 0.10, 0.07, 0.04, 0.03, 0.02]

SUPPLIER_NAMES = [
    "TechVision Industries",
    "Global Traders Pvt Ltd",
    "Quality Manufacturing Co.",
    "Sunrise Enterprises",
    "Elite Products Ltd",
    "Prime Suppliers",
    "Royal Merchants",
    "Shakti Trading",
    "Vertex Manufacturing",
    "Horizon Global",
    "Zenith Enterprises",
    "Nexus Supply Co.",
    "Apex Industries",
    "Summit Suppliers",
    "Crest Merchants",
    "Vanguard Trading",
    "Pinnacle Manufacturing",
    "Regal Products",
    "Nova Industries",
    "Orion Suppliers",
]

STORE_NAMES = [
    "Mumbai Flagship",
    "Delhi Superstore",
    "Bengaluru Central",
    "Chennai Plaza",
    "Hyderabad Hub",
    "Pune Retail",
    "Ahmedabad Mall",
    "Surat Centre",
    "Jaipur Junction",
    "Lucknow Point",
]

PAYMENT_METHODS = [
    ("Credit Card", 0.35),
    ("Debit Card", 0.25),
    ("UPI", 0.22),
    ("Net Banking", 0.10),
    ("Cash on Delivery", 0.06),
    ("Wallet", 0.02),
]

ORDER_STATUSES = [
    ("Delivered", 0.88),
    ("Cancelled", 0.05),
    ("Returned", 0.04),
    ("Processing", 0.02),
    ("Shipped", 0.01),
]

RETURN_REASONS = [
    "Defective Product",
    "Wrong Item Sent",
    "Quality Issue",
    "Size/Fit Issue",
    "Changed Mind",
    "Damaged in Transit",
    "Not as Described",
]

MARKETING_CHANNELS = [
    ("Google Ads", 0.28),
    ("Facebook Ads", 0.22),
    ("Instagram Ads", 0.18),
    ("YouTube Ads", 0.12),
    ("Email Marketing", 0.08),
    ("Affiliate", 0.06),
    ("SMS Marketing", 0.04),
    ("SEO Organic", 0.02),
]

DEVICE_TYPES = [
    ("Mobile", 0.62),
    ("Desktop", 0.28),
    ("Tablet", 0.10),
]

EMPLOYEE_ROLES = [
    "Store Manager",
    "Sales Associate",
    "Customer Service",
    "Warehouse Manager",
    "Inventory Clerk",
    "Marketing Specialist",
    "Finance Analyst",
    "Operations Manager",
]

FIRST_NAMES_M = [
    "Rahul", "Rohan", "Arjun", "Aditya", "Vihaan",
    "Aarav", "Vivaan", "Reyansh", "Krishna", "Ishaan", "Sai", "Shivansh",
    "Ritvik", "Siddharth", "Pratham", "Atharv", "Vedant", "Kabir",
    "Aryan", "Raghav", "Abhinav", "Ankit", "Ayush", "Sahil", "Varun",
    "Dev", "Dhruv", "Harsh", "Karan", "Mohit", "Nikhil", "Pranav",
    "Suresh", "Rajesh", "Amit", "Vikram", "Rakesh", "Manish", "Gaurav",
    "Sourabh", "Vaibhav", "Mayank", "Shubham", "Naman", "Anirudh", "Yash",
    "Aadi", "Zaid", "Amey", "Yuvraj", "Ankur", "Hardik", "Lakshya",
    "Madhav", "Rushil", "Samarth", "Tanay", "Vansh", "Arnav", "Kunal",
    "Sameer", "Rohit",
]

FIRST_NAMES_F = [
    "Ananya", "Aadhya", "Diya", "Aarohi", "Saanvi", "Anika", "Ishita",
    "Aarushi", "Pari", "Riya", "Sara", "Shreya", "Kiara", "Myra",
    "Aditi", "Siya", "Navya", "Vanya", "Anusha", "Kavya", "Riddhi",
    "Siddhi", "Tara", "Avni", "Rashmi", "Pooja", "Neha", "Priya",
    "Meera", "Ritu", "Sneha", "Swati", "Mansi", "Nisha", "Shalini",
    "Tanvi", "Disha", "Nandini", "Ruchi", "Shilpa", "Aishwarya", "Anjali",
    "Bhavana", "Chhavi", "Esha", "Fatima", "Gayatri", "Harshita", "Ira",
    "Jhanvi", "Kriti", "Lakshmi", "Mahi", "Nitya", "Roshni", "Saumya",
    "Trisha", "Akshara", "Vrinda",
]

LAST_NAMES = [
    "Kumar", "Singh", "Sharma", "Patel", "Iyer", "Reddy", "Nair", "Das",
    "Mukherjee", "Bose", "Verma", "Gupta", "Mehta", "Shah", "Kapoor",
    "Khan", "Ahmed", "Joshi", "Desai", "Rao", "Chauhan", "Yadav", "Rajput",
    "Tiwari", "Pandey", "Pawar", "Chaudhary", "Agarwal", "Mishra", "Bhat",
    "Sinha", "Ghosh", "Mazumdar", "Naidu", "Menon", "Saxena", "Tripathi",
    "Malhotra", "Sethi", "Khanna", "Malik", "Jain", "Vaswani", "Thakur",
    "Bhardwaj", "Chaturvedi", "Shukla", "Awasthi", "Dubey", "Chopra",
    "Banerjee", "Sarkar", "Hegde", "Kamath", "Pillai", "Balakrishnan",
]
