from open_redactor.pii import find_pii_in_text, luhn_ok
from open_redactor.presets import PRESETS


def test_luhn():
    assert luhn_ok("4242424242424242")  # well known test card number
    assert not luhn_ok("4242424242424241")
    assert not luhn_ok("1234567890123")


def test_find_card_ssn_email_phone():
    text = "Card 4242 4242 4242 4242, SSN 078-05-1120, mail test@example.com, call 415-555-2671"
    kinds = [h.kind for h in find_pii_in_text(text)]
    assert "credit card number" in kinds
    assert "ssn" in kinds
    assert "email" in kinds
    assert "phone number" in kinds


def test_no_false_card():
    hits = find_pii_in_text("Order 1234567890123456 shipped on 2026-10-05")
    assert all(h.kind != "credit card number" for h in hits)


def test_invalid_ssn_area():
    hits = find_pii_in_text("Number 666-05-1120 here")
    assert all(h.kind != "ssn" for h in hits)


def test_documents_preset():
    assert "passport" in PRESETS["documents"]["targets"]
    assert "credit card" in PRESETS["documents"]["targets"]
