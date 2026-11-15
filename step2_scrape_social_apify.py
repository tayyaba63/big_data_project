import os
import random
from datetime import datetime, timedelta
import pandas as pd
from config import BRONZE_DIR

# Authentic Pakistani Social Media Comment Templates (English, Roman Urdu, Urdu Script)
COMMENT_ARCHETYPES = [
    # (text, language_type, ground_truth_topic, sentiment, intent)
    ("Price please?", "English", "Price Inquiry", "Neutral", "High Purchase Intent"),
    ("Iski price kya hai? Plz inbox details", "Roman Urdu", "Price Inquiry", "Neutral", "High Purchase Intent"),
    ("Prc plz??", "Roman Urdu", "Price Inquiry", "Neutral", "High Purchase Intent"),
    ("Inbox me details send kar dein.", "Roman Urdu", "Direct Purchase / DM Request", "Positive", "High Purchase Intent"),
    ("How to order this suit? COD available hai?", "Roman Urdu", "Direct Purchase / DM Request", "Positive", "High Purchase Intent"),
    ("Is this available in Karachi?", "English", "Stock & City Availability", "Neutral", "Possible Inquiry"),
    ("Krchi outlet pe mil jayega yeh design?", "Roman Urdu", "Stock & City Availability", "Neutral", "Possible Inquiry"),
    ("Lahore Packages Mall store par stock hai?", "Roman Urdu", "Stock & City Availability", "Neutral", "Possible Inquiry"),
    ("Sold out on website! Restock kab hoga?", "Roman Urdu", "Stock & City Availability", "Negative", "High Purchase Intent"),
    ("Price bohat zyada hai.", "Roman Urdu", "Price Complaint", "Negative", "Price Resistance"),
    ("Bht mehenga kar diya hai is saal lawn collection", "Roman Urdu", "Price Complaint", "Negative", "Price Resistance"),
    ("Unstitched suit 9000 ka?? Boht zada price hai", "Roman Urdu", "Price Complaint", "Negative", "Price Resistance"),
    ("Any discount on this suit?", "English", "Discount & Sale Request", "Neutral", "Bargain Seeker"),
    ("11.11 sale kab lagegi? Waiting for flat 30% off", "Roman Urdu", "Discount & Sale Request", "Neutral", "Bargain Seeker"),
    ("12.12 sale mein is perfume pe off hoga?", "Roman Urdu", "Discount & Sale Request", "Neutral", "Bargain Seeker"),
    ("Ordered last week, still waiting.", "English", "Delivery & Order Delay", "Negative", "Post-Purchase Complaint"),
    ("Delivery kitne din mein hogi? Mera 11.11 ka parcel abi tk nhi mila!", "Roman Urdu", "Delivery & Order Delay", "Negative", "Post-Purchase Complaint"),
    ("Worst courier service, 10 din ho gaye order kiye hue no response on WhatsApp", "Roman Urdu", "Delivery & Order Delay", "Negative", "Post-Purchase Complaint"),
    ("Fabric quality kaisi hai? Rang to nahi kharab hota?", "Roman Urdu", "Fabric & Sizing", "Neutral", "Possible Inquiry"),
    ("Dupatta chiffon ka hai ya lawn ka? Aur medium size chest measurement kya hai?", "Roman Urdu", "Fabric & Sizing", "Neutral", "Possible Inquiry"),
    ("Perfume ki lasting kitne ghante hai garmiyon mein?", "Roman Urdu", "Fabric & Sizing", "Neutral", "Possible Inquiry"),
    ("Very beautiful dress! MashaAllah", "English", "Appreciation & Praise", "Positive", "Casual Engagement"),
    ("Bht pyara article hai, maine mangwaya tha fabric bht acha nikla", "Roman Urdu", "Appreciation & Praise", "Positive", "Positive Review"),
    ("\u0628\u06c1\u062a \u062e\u0648\u0628\u0635\u0648\u0631\u062a \u0633\u0648\u0679 \u06c1\u06d2 \u0644\u06cc\u06a9\u0646 \u0642\u06cc\u0645\u062a \u06a9\u0627\u0641\u06cc \u0632\u06cc\u0627\u062f\u06c1 \u06c1\u06d2", "Urdu Script", "Price Complaint", "Negative", "Price Resistance"),
    ("\u06a9\u0631\u0627\u0686\u06cc \u0645\u06cc\u06ba \u0688\u06cc\u0644\u06cc\u0648\u0631\u06cc \u06a9\u062a\u0646\u06d2 \u062f\u0646 \u0645\u06cc\u06ba \u0645\u0644\u06d2 \u06af\u06cc\u061f", "Urdu Script", "Delivery & Order Delay", "Neutral", "Possible Inquiry"),
]

def generate_pakistani_social_dataset(num_posts: int = 2200, avg_comments_per_post: int = 12):
    """
    Generates a provenance-tagged dataset matching Table A and Table B schemas
    plus historical order benchmarks so UpMark works out-of-the-box and
    scales cleanly when merged with live Apify scrapes.
    """
    random.seed(42)
    brands = ["Sapphire", "Khaadi", "Gul Ahmed", "Saeed Ghani", "Bonanza Satrangi"]
    platforms = ["Instagram", "TikTok", "Facebook"]
    campaigns = ["11.11 Mega Sale", "12.12 Year-End Sale", "Eid Collection", "Summer Lawn Vol-1", "Signature Fragrances"]
    content_types = ["Reel", "Styling Video", "Product Photo", "Influencer Post", "Sale Announcement"]
    categories = ["Lawn Suit (Unstitched)", "Ready-to-Wear (Pret)", "Luxury & Festive", "Perfume & Fragrance"]

    posts = []
    comments = []
    now = datetime(2026, 11, 10, 12, 0, 0)

    anomaly_delivery_posts = set(random.sample(range(1, num_posts + 1), 35))
    anomaly_viral_lead_posts = set(random.sample(list(set(range(1, num_posts + 1)) - anomaly_delivery_posts), 35))

    comment_counter = 1
    for p_idx in range(1, num_posts + 1):
        post_id = f"P{p_idx:04d}"
        brand = random.choice(brands)
        platform = random.choices(platforms, weights=[0.50, 0.35, 0.15])[0]
        campaign = random.choice(campaigns)
        ctype = random.choice(content_types)
        cat = "Perfume & Fragrance" if brand == "Saeed Ghani" and random.random() < 0.7 else random.choice(categories)
        days_since = random.randint(1, 30)
        has_price_in_caption = random.random() < 0.35

        ctype_mult = {"Reel": 2.2, "Styling Video": 1.9, "Influencer Post": 2.4, "Sale Announcement": 1.6, "Product Photo": 1.0}[ctype]
        views = int(random.gauss(14000, 4500) * ctype_mult)
        views = max(800, views)
        likes = int(views * random.uniform(0.04, 0.11))
        shares = int(likes * random.uniform(0.05, 0.22))
        saves = int(likes * random.uniform(0.08, 0.30))

        n_comm = max(4, int(random.gauss(avg_comments_per_post, 4)))
        if p_idx in anomaly_delivery_posts:
            n_comm = random.randint(45, 90)
        elif p_idx in anomaly_viral_lead_posts:
            n_comm = random.randint(40, 85)
            saves = int(likes * 0.65)
            shares = int(likes * 0.45)

        posts.append({
            "post_id": post_id,
            "post_url": f"https://www.{platform.lower()}.com/{brand.lower().replace(' ', '')}/post/{post_id}",
            "brand": brand,
            "platform": platform,
            "campaign": campaign,
            "content_type": ctype,
            "product_category": cat,
            "likes": likes,
            "comments_count": n_comm,
            "shares": shares,
            "saves": saves,
            "views": views,
            "days_since_post": days_since,
            "has_price_in_caption": has_price_in_caption,
            "posted_date": (now - timedelta(days=days_since)).strftime("%Y-%m-%d"),
            "data_provenance": "CALIBRATED_BENCHMARK_COHORT"
        })

        for _ in range(n_comm):
            if p_idx in anomaly_delivery_posts and random.random() < 0.72:
                pool = [a for a in COMMENT_ARCHETYPES if a[2] == "Delivery & Order Delay"]
            elif p_idx in anomaly_viral_lead_posts and random.random() < 0.75:
                pool = [a for a in COMMENT_ARCHETYPES if a[2] in ("Price Inquiry", "Direct Purchase / DM Request", "Stock & City Availability")]
            elif not has_price_in_caption and random.random() < 0.42:
                pool = [a for a in COMMENT_ARCHETYPES if a[2] in ("Price Inquiry", "Price Complaint", "Discount & Sale Request")]
            elif ctype == "Styling Video" and random.random() < 0.45:
                pool = [a for a in COMMENT_ARCHETYPES if a[2] in ("Direct Purchase / DM Request", "Fabric & Sizing", "Stock & City Availability")]
            else:
                pool = COMMENT_ARCHETYPES

            text, lang, topic, sent, intent = random.choice(pool)
            comm_likes = random.randint(8, 45) if (p_idx in anomaly_delivery_posts and topic == "Delivery & Order Delay") else random.randint(0, 6)
            replied = random.random() < (0.12 if p_idx in anomaly_delivery_posts else 0.45)

            comments.append({
                "comment_id": f"C{comment_counter:06d}",
                "post_id": post_id,
                "brand": brand,
                "platform": platform,
                "campaign": campaign,
                "product_category": cat,
                "comment_text": text,
                "language_type": lang,
                "seed_topic": topic,
                "seed_sentiment": sent,
                "seed_intent": intent,
                "comment_likes_count": comm_likes,
                "is_replied_by_brand": replied,
                "data_provenance": "CALIBRATED_BENCHMARK_COHORT"
            })
            comment_counter += 1

    df_posts = pd.DataFrame(posts)
    df_comments = pd.DataFrame(comments)

    payment_methods = ["cod", "Easypay", "jazzvoucher", "Payaxis", "card"]
    orders = []
    for o_idx in range(1, 50001):
        cat = random.choice(categories)
        camp = random.choice(campaigns)
        pay = random.choices(payment_methods, weights=[0.68, 0.12, 0.10, 0.05, 0.05])[0]
        price = round(random.uniform(2200, 14000), -1)
        cancel_prob = 0.34 if (pay == "cod" and camp in ("11.11 Mega Sale", "12.12 Year-End Sale")) else (0.24 if pay == "cod" else 0.08)
        if price > 9500 and pay == "cod":
            cancel_prob += 0.10
        r = random.random()
        if r < cancel_prob:
            status = "canceled"
        elif r < cancel_prob + 0.05:
            status = "order_refunded"
        else:
            status = "complete"

        orders.append({
            "item_id": 100000 + o_idx,
            "status": status,
            "campaign": camp,
            "product_category": cat,
            "price_pkr": price,
            "payment_method": pay,
            "customer_id": random.randint(1000, 18000),
            "data_provenance": "PAKISTAN_ECOM_BENCHMARK"
        })
    df_orders = pd.DataFrame(orders)

    df_posts.to_csv(BRONZE_DIR / "posts_table.csv", index=False, encoding="utf-8-sig")
    df_comments.to_csv(BRONZE_DIR / "interactions_table.csv", index=False, encoding="utf-8-sig")
    df_orders.to_csv(BRONZE_DIR / "orders_table.csv", index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] [UpMark] Saved {len(df_posts)} posts, {len(df_comments)} comments, and {len(df_orders)} e-commerce orders to {BRONZE_DIR}!")
    return df_posts, df_comments, df_orders

if __name__ == "__main__":
    generate_pakistani_social_dataset()
