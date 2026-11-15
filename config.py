import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
DUCKDB_PATH = DATA_DIR / "upmark.duckdb"

for d in [DATA_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 1. Public Shopify JSON Endpoints for Pakistani Clothing & Perfume Brands
BRAND_STORES = [
    {"brand": "Sapphire", "base_url": "https://pk.sapphireonline.pk/products.json?limit=250&page="},
    {"brand": "Saeed Ghani", "base_url": "https://saeedghani.pk/products.json?limit=250&page="},
    {"brand": "Bonanza Satrangi", "base_url": "https://bonanzasatrangi.com/products.json?limit=250&page="},
    {"brand": "Beechtree", "base_url": "https://beechtree.pk/products.json?limit=250&page="},
]

# 2. Roman Urdu Phonetic Canonicalization Dictionary (Resolves spelling variance)
ROMAN_URDU_LEXICON = {
    "bht": "bohat",
    "buhat": "bohat",
    "boht": "bohat",
    "bauhat": "bohat",
    "zyada": "ziyada",
    "zada": "ziyada",
    "ziada": "ziyada",
    "krchi": "karachi",
    "khi": "karachi",
    "lhr": "lahore",
    "isb": "islamabad",
    "rwp": "rawalpindi",
    "fsd": "faisalabad",
    "prc": "price",
    "pries": "price",
    "priz": "price",
    "kya": "kia",
    "plz": "please",
    "pls": "please",
    "plzz": "please",
    "dlvry": "delivery",
    "delivry": "delivery",
    "dunge": "denge",
    "kb": "kab",
    "tk": "tak",
    "abhi": "abhi",
    "abi": "abhi",
    "nhi": "nahi",
    "nai": "nahi",
    "ni": "nahi",
    "mila": "received",
    "mili": "received",
    "kapra": "fabric",
    "kapray": "fabric",
    "stich": "stitched",
    "unstich": "unstitched",
    "dm": "inbox",
    "msg": "inbox",
    "inbx": "inbox",
    "avl": "available",
    "avail": "available",
    "avb": "available",
    "disc": "discount",
    "off": "discount",
}

# 3. Keyword Rules for Hybrid Topic & Intent Labelling (Seeds LightGBM / Zero-Shot)
TOPIC_KEYWORDS = {
    "Price Inquiry": ["price", "keemat", "rate", "how much", "kitne ka", "cost", "pkr", "rs"],
    "Discount & Sale Request": ["discount", "sale", "11.11", "12.12", "promo", "offer", "code", "off"],
    "Price Complaint": ["bohat ziyada", "expensive", "mehenga", "mehnga", "overpriced", "high price", "afford"],
    "Delivery & Order Delay": ["delivery", "waiting", "ordered last week", "still waiting", "parcel", "courier", "late", "received", "rider", "tracking"],
    "Stock & City Availability": ["available", "karachi", "lahore", "islamabad", "outlets", "store", "in stock", "restock", "sold out"],
    "Fabric & Sizing": ["fabric", "size", "sizing", "lawn", "chiffon", "dupatta", "quality", "khaddar", "stitched", "unstitched", "lasting", "fragrance"],
    "Direct Purchase / DM Request": ["inbox", "details send", "order karna", "link", "whatsapp", "book", "want to buy", "cod"],
}
