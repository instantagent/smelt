#!/usr/bin/env python3
"""Deterministic generator for the structured-task quality canary corpus.

Emits tools/quality-canary/structured-corpus.jsonl: 50 JSON-extraction /
tool-call-shaped cases. Fully deterministic (no RNG, no network) so the
committed .jsonl is reproducible byte-for-byte from this script.

Each case: {"id","category","text","fields":[...],"target":{...}}
`fields` is the ordered list of keys the model must emit; `target` is the
ground-truth value for every field. All target values appear verbatim (or as
plain numbers) in `text`, so exact/normalized matching is a fair "did it read
the input" test with no reasoning cliff -- this is a canary, not a reasoning
bench. The runner applies one fixed neutral prompt; no per-case tuning.
"""
import json
import sys

cases = []


def add(cid, category, text, target):
    cases.append({"id": cid, "category": category, "text": text,
                  "fields": list(target.keys()), "target": target})


contacts = [
    ("Reach Dana Whitfield at dana.whitfield@example.com or 415-555-0182.",
     {"name": "Dana Whitfield", "email": "dana.whitfield@example.com", "phone": "415-555-0182"}),
    ("Please CC marcus_lee@corp.io on the thread; his desk line is 206-555-0117.",
     {"name": "Marcus Lee", "email": "marcus_lee@corp.io", "phone": "206-555-0117"}),
    ("Contact: Priya Raman, priya.raman@mail.net, +1-312-555-0143.",
     {"name": "Priya Raman", "email": "priya.raman@mail.net", "phone": "+1-312-555-0143"}),
    ("Our new hire Tomas Okafor (tomas.okafor@team.dev) can be reached at 646-555-0190.",
     {"name": "Tomas Okafor", "email": "tomas.okafor@team.dev", "phone": "646-555-0190"}),
    ("If urgent, call Helen Castro on 503-555-0166; email helen.castro@shop.co.",
     {"name": "Helen Castro", "email": "helen.castro@shop.co", "phone": "503-555-0166"}),
    ("Sales lead: wei.zhang@bizmail.com. Ask for Wei Zhang at 702-555-0128.",
     {"name": "Wei Zhang", "email": "wei.zhang@bizmail.com", "phone": "702-555-0128"}),
    ("Send the invoice to Fatima Noor, fatima.noor@ledger.org, phone 858-555-0173.",
     {"name": "Fatima Noor", "email": "fatima.noor@ledger.org", "phone": "858-555-0173"}),
    ("Booking held under Greg Sullivan; confirm at greg.sullivan@stay.travel / 917-555-0155.",
     {"name": "Greg Sullivan", "email": "greg.sullivan@stay.travel", "phone": "917-555-0155"}),
    ("The account manager is Aisha Bello (aisha.bello@fund.io), direct 214-555-0139.",
     {"name": "Aisha Bello", "email": "aisha.bello@fund.io", "phone": "214-555-0139"}),
    ("Ping Oliver Grant at oliver.grant@labs.ai or his cell 480-555-0121.",
     {"name": "Oliver Grant", "email": "oliver.grant@labs.ai", "phone": "480-555-0121"}),
    ("Escalations go to Nina Petrova, nina.petrova@ops.net, 617-555-0148.",
     {"name": "Nina Petrova", "email": "nina.petrova@ops.net", "phone": "617-555-0148"}),
    ("For press, email sam.robinson@news.press; Sam Robinson answers at 202-555-0164.",
     {"name": "Sam Robinson", "email": "sam.robinson@news.press", "phone": "202-555-0164"}),
]
for i, (t, tgt) in enumerate(contacts, 1):
    add("contact-%02d" % i, "contact", t, tgt)

orders = [
    ("Order confirmed: 3 units of Wireless Mouse at 24.99 USD each.",
     {"product": "Wireless Mouse", "quantity": 3, "unit_price": 24.99, "currency": "USD"}),
    ("The customer bought 12 Ceramic Mug items for 8.50 EUR apiece.",
     {"product": "Ceramic Mug", "quantity": 12, "unit_price": 8.50, "currency": "EUR"}),
    ("Cart: 1 Standing Desk, unit price 349.00 USD.",
     {"product": "Standing Desk", "quantity": 1, "unit_price": 349.00, "currency": "USD"}),
    ("Shipped 5 boxes of Trail Mix at 6.25 GBP each.",
     {"product": "Trail Mix", "quantity": 5, "unit_price": 6.25, "currency": "GBP"}),
    ("Invoice line: Bluetooth Speaker x2, 59.95 USD per unit.",
     {"product": "Bluetooth Speaker", "quantity": 2, "unit_price": 59.95, "currency": "USD"}),
    ("Purchase: 20 Notebook units priced at 3.75 CAD each.",
     {"product": "Notebook", "quantity": 20, "unit_price": 3.75, "currency": "CAD"}),
    ("Reserved 4 Yoga Mat items at 18.00 AUD apiece.",
     {"product": "Yoga Mat", "quantity": 4, "unit_price": 18.00, "currency": "AUD"}),
    ("Order 88: 7 LED Bulb, 2.40 USD each.",
     {"product": "LED Bulb", "quantity": 7, "unit_price": 2.40, "currency": "USD"}),
    ("Restock request: 15 Coffee Filter packs at 4.10 EUR per pack.",
     {"product": "Coffee Filter", "quantity": 15, "unit_price": 4.10, "currency": "EUR"}),
    ("The buyer selected 2 Desk Lamp units, 41.50 USD each.",
     {"product": "Desk Lamp", "quantity": 2, "unit_price": 41.50, "currency": "USD"}),
]
for i, (t, tgt) in enumerate(orders, 1):
    add("order-%02d" % i, "order", t, tgt)

events = [
    ("The Design Review is scheduled for 2026-03-14 in Conference Room B.",
     {"title": "Design Review", "date": "2026-03-14", "location": "Conference Room B"}),
    ("Join the Quarterly Planning meeting on 2026-01-09 at the Downtown Office.",
     {"title": "Quarterly Planning", "date": "2026-01-09", "location": "Downtown Office"}),
    ("Onboarding Session happens 2026-02-02 in the Training Lab.",
     {"title": "Onboarding Session", "date": "2026-02-02", "location": "Training Lab"}),
    ("Save the date: Product Launch on 2026-05-20 at the Main Auditorium.",
     {"title": "Product Launch", "date": "2026-05-20", "location": "Main Auditorium"}),
    ("The Budget Sync is set for 2026-04-11 in Room 402.",
     {"title": "Budget Sync", "date": "2026-04-11", "location": "Room 402"}),
    ("Team Offsite takes place 2026-06-18 at the Lakeside Lodge.",
     {"title": "Team Offsite", "date": "2026-06-18", "location": "Lakeside Lodge"}),
    ("Security Training is on 2026-03-27 in the East Wing.",
     {"title": "Security Training", "date": "2026-03-27", "location": "East Wing"}),
    ("The Hiring Panel convenes 2026-02-15 in Interview Room 3.",
     {"title": "Hiring Panel", "date": "2026-02-15", "location": "Interview Room 3"}),
    ("Customer Demo scheduled 2026-07-01 at the Client Center.",
     {"title": "Customer Demo", "date": "2026-07-01", "location": "Client Center"}),
    ("The Retro meeting is 2026-01-31 in the Blue Room.",
     {"title": "Retro", "date": "2026-01-31", "location": "Blue Room"}),
]
for i, (t, tgt) in enumerate(events, 1):
    add("event-%02d" % i, "event", t, tgt)

bios = [
    ("Elena Novak is a 34-year-old architect based in Prague.",
     {"name": "Elena Novak", "age": 34, "occupation": "architect", "city": "Prague"}),
    ("Meet Rajesh Kumar, a 41-year-old surgeon from Mumbai.",
     {"name": "Rajesh Kumar", "age": 41, "occupation": "surgeon", "city": "Mumbai"}),
    ("Clara Mendes, 29, works as a journalist in Lisbon.",
     {"name": "Clara Mendes", "age": 29, "occupation": "journalist", "city": "Lisbon"}),
    ("Yuki Tanaka is a 52-year-old chef living in Osaka.",
     {"name": "Yuki Tanaka", "age": 52, "occupation": "chef", "city": "Osaka"}),
    ("David Owens, a 38-year-old pilot, resides in Denver.",
     {"name": "David Owens", "age": 38, "occupation": "pilot", "city": "Denver"}),
    ("Amara Diallo is a 26-year-old biologist based in Dakar.",
     {"name": "Amara Diallo", "age": 26, "occupation": "biologist", "city": "Dakar"}),
    ("Lucas Ferreira, 47, is a teacher in Porto.",
     {"name": "Lucas Ferreira", "age": 47, "occupation": "teacher", "city": "Porto"}),
    ("Sofia Ricci is a 31-year-old economist from Milan.",
     {"name": "Sofia Ricci", "age": 31, "occupation": "economist", "city": "Milan"}),
]
for i, (t, tgt) in enumerate(bios, 1):
    add("bio-%02d" % i, "bio", t, tgt)

tools = [
    ("What's the weather in Seattle right now?",
     {"name": "get_weather", "arguments": {"location": "Seattle"}}),
    ("Set a timer for 10 minutes.",
     {"name": "set_timer", "arguments": {"minutes": 10}}),
    ("Play the song Clocks by Coldplay.",
     {"name": "play_song", "arguments": {"title": "Clocks", "artist": "Coldplay"}}),
    ("Add milk to my shopping list.",
     {"name": "add_to_list", "arguments": {"item": "milk"}}),
    ("Translate hello into French.",
     {"name": "translate", "arguments": {"text": "hello", "target_language": "French"}}),
    ("Convert 100 USD to EUR.",
     {"name": "convert_currency", "arguments": {"amount": 100, "from": "USD", "to": "EUR"}}),
    ("Remind me to call the dentist at 3pm.",
     {"name": "set_reminder", "arguments": {"task": "call the dentist", "time": "3pm"}}),
    ("Turn the living room lights off.",
     {"name": "control_lights", "arguments": {"room": "living room", "state": "off"}}),
    ("Search for pasta recipes.",
     {"name": "web_search", "arguments": {"query": "pasta recipes"}}),
    ("Book a table for 4 at Nobu.",
     {"name": "book_table", "arguments": {"party_size": 4, "restaurant": "Nobu"}}),
]
for i, (t, tgt) in enumerate(tools, 1):
    add("tool-%02d" % i, "tool_call", t, tgt)


def main():
    assert len(cases) == 50, "expected 50 cases, got %d" % len(cases)
    assert len(set(c["id"] for c in cases)) == 50, "duplicate case ids"
    out = sys.argv[1] if len(sys.argv) > 1 else "tools/quality-canary/structured-corpus.jsonl"
    with open(out, "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=True, sort_keys=True) + "\n")
    print("wrote %d cases to %s" % (len(cases), out))


if __name__ == "__main__":
    main()
