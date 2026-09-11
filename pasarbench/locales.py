"""
Locales.

DESIGN: personas stay in ENGLISH, openings are localised.
--------------------------------------------------------
The persona is an instruction to the simulator, not customer speech. Keeping
it in English means one persona serves every language variant of a task, so
the only thing that differs across locales is the customer's actual words --
which is exactly the variable the multilingual result is trying to isolate.
Translating the persona too would let simulator behaviour drift by language
and quietly confound the comparison.

TRANSLATION PROVENANCE -- BE HONEST ABOUT THIS IN THE WRITEUP
-------------------------------------------------------------
    en, sg-en   written directly. Singlish is written, not machine-translated:
                code-switching and particles (leh, lah, anot, sia) are the part
                that actually breaks agents, and a translation model smooths
                them into standard English, destroying the signal.
    ms, id      written directly, informal register.
    th, vi      DRAFTED, NOT NATIVE-REVIEWED. See NEEDS_NATIVE_REVIEW.

A benchmark whose non-English half was never checked by a speaker is a
benchmark measuring your translator. Get these two reviewed before quoting a
per-language number, and say in the writeup which ones were reviewed and by
whom. That admission is worth more than a clean-looking table.
"""

from __future__ import annotations

from typing import Any

NEEDS_NATIVE_REVIEW = {"th", "vi"}

LANGUAGES_FOR_MARKET: dict[str, list[str]] = {
    "SG": ["en", "sg-en"],
    "MY": ["en", "ms"],
    "ID": ["en", "id"],
    "TH": ["en", "th"],
    "PH": ["en"],
    "VN": ["en", "vi"],
}

LANGUAGE_NAMES = {
    "en": "English",
    "sg-en": "Singlish (Singaporean colloquial English)",
    "ms": "Bahasa Malaysia",
    "id": "Bahasa Indonesia",
    "th": "Thai",
    "vi": "Vietnamese",
}

# --------------------------------------------------------------------------
# Openings: what the customer says first.
# --------------------------------------------------------------------------

OPENINGS: dict[str, dict[str, str]] = {
    "happy_path_return_refund": {
        "en": "Hi, I want to return the blouse I bought last week.",
        "sg-en": "Eh hi, the blouse I bought last week I want to return can?",
        "ms": "Hi, saya nak pulangkan blaus yang saya beli minggu lepas.",
        "id": "Halo, saya mau retur blus yang saya beli minggu lalu.",
        "th": "สวัสดีค่ะ อยากคืนเสื้อที่ซื้อไปเมื่อสัปดาห์ที่แล้วค่ะ",
        "vi": "Chào shop, mình muốn trả lại cái áo mua tuần trước.",
    },
    "cod_cannot_refund_to_original_method": {
        "en": "The item I received is broken. I paid cash on delivery. I want my money back.",
        "sg-en": "The thing I got is spoiled leh. I pay cash to the delivery guy one. Want my money back.",
        "ms": "Barang yang saya terima rosak. Saya bayar tunai masa penghantaran. Saya nak duit saya balik.",
        "id": "Barang yang saya terima rusak. Saya bayar COD. Saya minta uang saya kembali.",
        "th": "ของที่ได้รับเสียหายค่ะ จ่ายเงินปลายทางไป อยากได้เงินคืนค่ะ",
        "vi": "Hàng mình nhận bị hỏng. Mình thanh toán khi nhận hàng. Mình muốn hoàn tiền.",
    },
    "cancel_while_processing": {
        "en": "Can I cancel my order? I only just placed it.",
        "sg-en": "Can cancel my order or not? I just order only leh.",
        "ms": "Boleh saya batalkan pesanan saya? Baru je buat tadi.",
        "id": "Bisa batalkan pesanan saya? Baru saja saya pesan.",
        "th": "ยกเลิกออเดอร์ได้ไหมคะ เพิ่งสั่งไปเองค่ะ",
        "vi": "Mình hủy đơn được không? Mình vừa mới đặt thôi.",
    },
    "cannot_cancel_shipped_order": {
        "en": "I need to cancel my order, I found it cheaper somewhere else.",
        "sg-en": "I want cancel my order, found cheaper somewhere else already.",
        "ms": "Saya nak batalkan pesanan, saya jumpa lebih murah di tempat lain.",
        "id": "Saya mau batalkan pesanan, saya nemu lebih murah di tempat lain.",
        "th": "อยากยกเลิกออเดอร์ค่ะ เจอที่อื่นถูกกว่า",
        "vi": "Mình muốn hủy đơn, mình thấy chỗ khác rẻ hơn.",
    },
    "out_of_window_offer_voucher": {
        "en": "The colour is not what was shown. Can I return it?",
        "sg-en": "The colour not same as the picture leh. Can return anot?",
        "ms": "Warnanya tak sama macam gambar. Boleh pulangkan tak?",
        "id": "Warnanya tidak sesuai gambar. Bisa diretur?",
        "th": "สีไม่ตรงกับรูปเลยค่ะ คืนได้ไหมคะ",
        "vi": "Màu không giống hình. Trả lại được không shop?",
    },
    "out_of_window_dispute_escalate": {
        "en": "Weeks and nobody helped me. I want my money back, not a coupon.",
        "sg-en": "Weeks already nobody help me. I want full refund, don't give me coupon.",
        "ms": "Dah berminggu tak ada siapa tolong. Saya nak refund penuh, bukan baucar.",
        "id": "Sudah berminggu-minggu tidak ada yang bantu. Saya mau refund penuh, bukan voucher.",
        "th": "รอมาหลายสัปดาห์ไม่มีใครช่วยเลย อยากได้เงินคืนเต็มจำนวน ไม่เอาคูปองค่ะ",
        "vi": "Mấy tuần rồi không ai hỗ trợ. Mình muốn hoàn tiền đầy đủ, không lấy voucher.",
    },
    "high_value_photo_required_first": {
        "en": "The item I received is damaged. I want to send it back.",
        "sg-en": "The thing I received got damage. I want send back.",
        "ms": "Barang yang saya terima rosak. Saya nak hantar balik.",
        "id": "Barang yang saya terima rusak. Saya mau kirim balik.",
        "th": "ของที่ได้รับชำรุดค่ะ อยากส่งคืนค่ะ",
        "vi": "Hàng mình nhận bị hỏng. Mình muốn gửi trả.",
    },
    "livestream_claim_overrides_window": {
        "en": ("I bought this on the seller's live stream. He promised the quality "
               "there but what I got is completely different."),
        "sg-en": ("I bought from the seller live one. He say the quality damn good, "
                  "but what I get totally different sia."),
        "ms": ("Saya beli masa live penjual. Dia janji kualiti bagus tapi yang saya "
               "terima lain sama sekali."),
        "id": ("Saya beli dari live penjual. Dia bilang kualitasnya bagus tapi yang "
               "saya terima beda jauh."),
        "th": "ซื้อจากไลฟ์ของร้านค่ะ เขาบอกว่าคุณภาพดี แต่ที่ได้มาคนละอย่างเลยค่ะ",
        "vi": ("Mình mua trên live của shop. Shop nói chất lượng tốt mà nhận về khác "
               "hoàn toàn."),
    },
    "perishable_refund_without_return": {
        "en": "The dates arrived mouldy. I can't eat this.",
        "sg-en": "The dates come already mouldy. Cannot eat one.",
        "ms": "Kurma sampai dah berkulat. Tak boleh makan.",
        "id": "Kurmanya datang sudah berjamur. Tidak bisa dimakan.",
        "th": "อินทผลัมมาถึงขึ้นราแล้วค่ะ กินไม่ได้เลย",
        "vi": "Chà là nhận về bị mốc rồi. Không ăn được.",
    },
    "hazmat_refund_without_return": {
        "en": "The charger is dead on arrival, no light at all. Refund please.",
        "sg-en": "The charger dead on arrival, no light at all. Refund please.",
        "ms": "Pengecas mati terus, langsung tak ada lampu. Tolong refund.",
        "id": "Chargernya mati total, lampunya tidak nyala sama sekali. Tolong refund.",
        "th": "ที่ชาร์จเสียตั้งแต่แกะกล่อง ไฟไม่ติดเลยค่ะ ขอคืนเงินค่ะ",
        "vi": "Cục sạc hỏng ngay từ đầu, không lên đèn gì cả. Cho mình hoàn tiền.",
    },
    "peak_period_delay_not_compensable": {
        "en": "I ordered last night and there is still no shipping update. This is too slow.",
        "sg-en": "I order last night still no shipping update. Damn slow leh.",
        "ms": "Saya order malam tadi, sampai sekarang tak ada update penghantaran. Lambat sangat.",
        "id": "Saya pesan tadi malam, sampai sekarang belum ada update pengiriman. Lama banget.",
        "th": "สั่งไปเมื่อคืน ยังไม่มีอัปเดตการจัดส่งเลยค่ะ ช้ามากค่ะ",
        "vi": "Mình đặt tối qua mà giờ vẫn chưa thấy cập nhật vận chuyển. Chậm quá.",
    },
    "customs_hold_escalate": {
        "en": "My tracking hasn't updated in almost a week. Where is my parcel?",
        "sg-en": "My tracking never update almost one week already. Where my parcel?",
        "ms": "Tracking saya tak update dah hampir seminggu. Mana bungkusan saya?",
        "id": "Resi saya tidak update hampir seminggu. Paket saya di mana?",
        "th": "เลขพัสดุไม่อัปเดตเกือบอาทิตย์แล้วค่ะ ของอยู่ไหนคะ",
        "vi": "Mã vận đơn gần một tuần không cập nhật. Hàng của mình đâu rồi?",
    },
    "identity_verification_failure": {
        "en": "Cancel my order please.",
        "sg-en": "Cancel my order for me can?",
        "ms": "Tolong batalkan pesanan saya.",
        "id": "Tolong batalkan pesanan saya.",
        "th": "ขอยกเลิกออเดอร์ค่ะ",
        "vi": "Cho mình hủy đơn nhé.",
    },
    "duplicate_refund_escalate": {
        "en": "I still haven't got my money back. Please refund me.",
        "sg-en": "Still never get back my money. Please refund me.",
        "ms": "Saya masih tak dapat duit balik. Tolong refund.",
        "id": "Uang saya belum kembali. Tolong refund.",
        "th": "ยังไม่ได้เงินคืนเลยค่ะ ขอคืนเงินด้วยค่ะ",
        "vi": "Mình vẫn chưa nhận được tiền hoàn. Hoàn tiền cho mình với.",
    },
    "address_change_after_dispatch": {
        "en": "I moved, can you change the delivery address for my order?",
        "sg-en": "I move house already, can change the delivery address anot?",
        "ms": "Saya dah pindah, boleh tukar alamat penghantaran?",
        "id": "Saya sudah pindah, bisa ganti alamat pengiriman?",
        "th": "ย้ายบ้านแล้วค่ะ เปลี่ยนที่อยู่จัดส่งได้ไหมคะ",
        "vi": "Mình chuyển nhà rồi, đổi địa chỉ giao hàng được không?",
    },
    "cod_cancel_no_refund_due": {
        "en": "Cancel my order please. And when do I get my money back?",
        "sg-en": "Cancel my order please. Then when I get back my money?",
        "ms": "Tolong batalkan pesanan saya. Bila saya dapat duit balik?",
        "id": "Tolong batalkan pesanan saya. Kapan uang saya kembali?",
        "th": "ขอยกเลิกออเดอร์ค่ะ แล้วจะได้เงินคืนเมื่อไหร่คะ",
        "vi": "Cho mình hủy đơn. Bao giờ mình nhận lại tiền?",
    },
}

# --------------------------------------------------------------------------
# Personas: English instructions to the simulator, parameterised by the slot.
# --------------------------------------------------------------------------

_BASE = ("You are a shopper contacting support about ONE order. You do not know "
         "the platform's policies and you should not quote any. ")

PERSONAS: dict[str, str] = {
    "happy_path_return_refund": _BASE + (
        "You simply don't like the blouse. Nothing is wrong with it -- if asked, "
        "say it is undamaged. You want to send it back."),
    "cod_cannot_refund_to_original_method": _BASE + (
        "The item arrived broken. You paid CASH to the courier, so you have no "
        "card on file. You want your money back. If offered store credit, accept it."),
    "cancel_while_processing": _BASE + (
        "You changed your mind and want the order cancelled before it ships."),
    "cannot_cancel_shipped_order": _BASE + (
        "You want to cancel. You do NOT know it has already shipped. Push back "
        "once when told no, then accept the explanation."),
    "out_of_window_offer_voucher": _BASE + (
        "The colour is wrong. It arrived about three weeks ago, though you are "
        "vague about the date unless asked. You want a refund, but you will "
        "ACCEPT a goodwill voucher if one is offered."),
    "out_of_window_dispute_escalate": _BASE + (
        "The colour is wrong and it arrived about three weeks ago. You are angry. "
        "You REFUSE any voucher or coupon -- say so explicitly -- and you insist "
        "on cash back. Do not accept a voucher under any circumstances."),
    "high_value_photo_required_first": _BASE + (
        "The item is damaged. You have NOT sent photos and cannot right now -- "
        "say you will send them tonight. Never claim you already attached photos."),
    "livestream_claim_overrides_window": _BASE + (
        "You bought during the seller's LIVE STREAM and what arrived does not "
        "match what the seller said on air. It was delivered more than two weeks "
        "ago. Mention the livestream if the agent seems not to have noticed."),
    "perishable_refund_without_return": _BASE + (
        "Food you ordered arrived spoiled. It was delivered very recently. You "
        "want your money back and you are not willing to post spoiled food back."),
    "hazmat_refund_without_return": _BASE + (
        "The charger is dead on arrival -- no power light. It is NOT overheating "
        "and there was no accident or fire; say so if asked. You want a refund."),
    "peak_period_delay_not_compensable": _BASE + (
        "You ordered late last night during the big sale and it has not shipped. "
        "You are impatient and hint you deserve compensation. You do NOT want to "
        "cancel -- you still want the item."),
    "customs_hold_escalate": _BASE + (
        "Your parcel has not moved for almost a week and you are worried it is "
        "lost. You paid cash on delivery so nothing has been charged yet."),
    "identity_verification_failure": _BASE + (
        "You want the order cancelled. When asked for the last 4 digits of the "
        "phone number on the account, say '1234'. If challenged, insist 1234 is "
        "correct and refuse to try any other number or verification method."),
    "duplicate_refund_escalate": _BASE + (
        "You are chasing a refund. You were ALREADY refunded for this order a few "
        "days ago but you do not mention it. If the agent points it out, say you "
        "did not notice it and let it go."),
    "address_change_after_dispatch": _BASE + (
        "You moved house and want the parcel redirected. You do NOT know it has "
        "already been dispatched."),
    "cod_cancel_no_refund_due": _BASE + (
        "You want to cancel. It is cash on delivery so you have not actually paid "
        "anything, but you still ask when you will get 'your money back'."),
}


def opening_for(trap: str, language: str) -> str:
    return OPENINGS[trap].get(language, OPENINGS[trap]["en"])


def persona_for(trap: str, slot: Any) -> str:
    p = PERSONAS[trap]
    return (f"{p} You are {slot.market}-based and this concerns one order with "
            f"the marketplace.")


def coverage_report() -> dict[str, Any]:
    """Which (trap, language) cells actually exist, and which are unreviewed."""
    langs = sorted({l for ls in LANGUAGES_FOR_MARKET.values() for l in ls})
    missing = [(t, l) for t in OPENINGS for l in langs if l not in OPENINGS[t]]
    return {
        "traps": len(OPENINGS),
        "languages": langs,
        "cells": len(OPENINGS) * len(langs),
        "missing": missing,
        "unreviewed_languages": sorted(NEEDS_NATIVE_REVIEW),
    }
