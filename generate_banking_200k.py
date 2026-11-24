"""
Massive Scale Banking Intent Dataset Generator (200k+ utterances across 110 intents).
Generates 220,000 high-quality, linguistically diverse banking queries spanning
8 financial domains, complete with slot variations, realistic customer registers,
urgency levels, and domain-specific terminology.
"""

import os
import random
import json
import itertools
from pathlib import Path
import pandas as pd
import numpy as np

# Seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# 1. Linguistic Slot Dictionaries for Dynamic Variation
# --------------------------------------------------------------------------
SLOTS = {
    "amounts": ["$50", "$120", "$250.50", "£40", "€100", "$500", "$1,200", "$5,000", "$10,000", "50 bucks", "a few hundred dollars"],
    "currencies": ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF"],
    "merchants": ["Amazon", "Uber", "Walmart", "Netflix", "Shell Gas Station", "Target", "Starbucks", "Airbnb", "Apple Store", "unrecognized vendor", "local grocery store"],
    "card_types": ["debit card", "credit card", "mastercard", "visa card", "black metal card", "virtual card", "contactless card", "ATM card"],
    "timeframes": ["yesterday", "this morning", "last Friday", "two days ago", "just now", "over the weekend", "last month"],
    "relatives": ["my brother", "my sister", "my landlord", "my friend", "my mom", "the contractor", "my business partner"],
    "countries": ["Germany", "the UK", "Japan", "France", "Spain", "Canada", "Australia", "Singapore", "India"],
    "devices": ["iPhone 15", "Samsung Galaxy", "MacBook Pro", "work laptop", "iPad", "Android phone"],
    "reasons": ["emergency travel", "buying a car", "paying rent", "medical bill", "tuition fees", "holiday booking", "home renovation"],
}

# --------------------------------------------------------------------------
# 2. Comprehensive Taxonomy of 110 Banking Intents across 8 Domains
# --------------------------------------------------------------------------
INTENT_SPECS = {
    # ---------------- Domain 1: Card Operations & Controls (16 Intents) ----------------
    "card_lost_or_stolen": {
        "domain": "Cards",
        "description": "Report lost, stolen, or misplaced physical card",
        "templates": [
            "I {lost} my {card_type} {timeframe} and need to cancel it right now",
            "Someone stole my wallet with my {card_type} inside, please help",
            "I left my {card_type} at a restaurant and can't find it anywhere",
            "My {card_type} went missing, how do I report it as lost",
            "Please deactivate my {card_type} immediately, it was stolen",
            "I suspect my {card_type} was picked from my pocket {timeframe}",
            "Urgent: my {card_type} has been stolen, freeze all transactions",
            "Lost my card while traveling in {countries}, what should I do",
        ],
        "substitutions": {"lost": ["lost", "misplaced", "dropped", "forgot", "can't find"]},
    },
    "card_freeze_unfreeze": {
        "domain": "Cards",
        "description": "Temporarily lock or unlock card controls",
        "templates": [
            "How can I temporarily {action} my {card_type}",
            "I want to {action} my {card_type} in the mobile app",
            "Can I toggle the security freeze on my {card_type} for a few hours",
            "Found my card! How do I unfreeze and resume spending on my {card_type}",
            "Please lock my {card_type} temporarily while I look for it",
            "My card is currently locked, how do I reactivate it to buy something",
            "Lock my {card_type} for safety reasons until further notice",
        ],
        "substitutions": {"action": ["freeze", "lock", "pause", "unfreeze", "unlock", "reactivate"]},
    },
    "card_damaged_replacement": {
        "domain": "Cards",
        "description": "Request replacement for broken, cracked, or worn card",
        "templates": [
            "My {card_type} chip is broken and contactless stopped working, send replacement",
            "The magnetic strip on my {card_type} is scratched, I need a new card",
            "My {card_type} cracked in half in my wallet, how do I order a new one",
            "Physical card is severely damaged, please order a replacement card",
            "My contactless chip is failing at payment terminals, replace my card",
            "Need to get a new physical card issued because the current one is unreadable",
        ],
        "substitutions": {},
    },
    "card_pin_change_reset": {
        "domain": "Cards",
        "description": "Change or reset payment card 4-digit PIN",
        "templates": [
            "I forgot the PIN for my {card_type}, how can I reset it",
            "Where in the app can I change my 4-digit card PIN",
            "I need to update my debit card security PIN to a new number",
            "Forgot my secret PIN code, please send a reset link",
            "How do I view or change my existing card PIN securely",
            "Can I set my card PIN online without visiting an ATM",
        ],
        "substitutions": {},
    },
    "card_pin_blocked": {
        "domain": "Cards",
        "description": "Unblock card after entering wrong PIN 3 times",
        "templates": [
            "My PIN has been blocked after entering it incorrectly three times",
            "Entered wrong PIN at the checkout and now my {card_type} is locked",
            "PIN blocked at ATM machine, how do I get it unblocked today",
            "How long does a PIN block last, or can I unblock it in the mobile app",
            "Card locked out due to multiple invalid PIN attempts, please unlock",
        ],
        "substitutions": {},
    },
    "card_activation": {
        "domain": "Cards",
        "description": "Activate newly received debit or credit card",
        "templates": [
            "My new {card_type} just arrived in the mail, how do I activate it",
            "Steps to activate my replacement physical card before first use",
            "I have my new card in hand, where do I enter the CVV to activate",
            "Need assistance activating my new contactless credit card",
            "Why is my card declining? Did I forget to complete card activation",
        ],
        "substitutions": {},
    },
    "card_delivery_status": {
        "domain": "Cards",
        "description": "Track postal shipping and delivery estimate for physical card",
        "templates": [
            "When will my new {card_type} arrive in the mail",
            "Can I get a tracking number for my replacement card shipment",
            "It has been 7 business days and my card hasn't arrived yet",
            "How long does standard delivery take for a new debit card",
            "Where is my card? Check delivery dispatch status please",
        ],
        "substitutions": {},
    },
    "card_virtual_disposable_create": {
        "domain": "Cards",
        "description": "Generate single-use virtual card for online purchases",
        "templates": [
            "How do I create a disposable virtual card for a one-time purchase",
            "Can I generate a virtual card number while waiting for physical card",
            "Where do I find my virtual card details in the mobile app",
            "I want a single-use card number to buy safely on {merchants}",
            "How many disposable virtual cards can I generate per day",
        ],
        "substitutions": {},
    },
    "card_contactless_limit_adjust": {
        "domain": "Cards",
        "description": "Adjust or enable/disable contactless tap-to-pay transaction limits",
        "templates": [
            "How do I increase my contactless payment limit from {amounts}",
            "Can I turn off contactless tap payments on my {card_type} for security",
            "Where can I change the maximum daily tap-to-pay threshold",
            "My contactless tap was declined because of the cumulative limit, reset it",
            "Adjust contactless limit for spending up to {amounts} per transaction",
        ],
        "substitutions": {},
    },
    "card_atm_swallowed": {
        "domain": "Cards",
        "description": "Card retained or swallowed by ATM machine",
        "templates": [
            "An ATM machine just swallowed my {card_type} and won't return it",
            "The cash machine took my card and displayed an error message",
            "My {card_type} was retained by a {merchants} ATM, what do I do",
            "ATM swallowed my card after cash withdrawal, can I retrieve it from the branch",
            "Card captured by cash machine, please cancel it immediately and reissue",
        ],
        "substitutions": {},
    },
    "card_expiration_renewal": {
        "domain": "Cards",
        "description": "Inquire about card expiry date and automatic renewal",
        "templates": [
            "My {card_type} is expiring at the end of this month, will you send a new one",
            "How early do you mail the automatic renewal card before expiry date",
            "My card expired {timeframe}, can I still use virtual card details",
            "Check if my renewal debit card has already been dispatched",
        ],
        "substitutions": {},
    },
    "card_international_travel_notice": {
        "domain": "Cards",
        "description": "Notify bank of international travel to avoid fraud blocks",
        "templates": [
            "I am traveling to {countries} next week, do I need to set a travel notice",
            "Going abroad to {countries}, please ensure my card isn't blocked for fraud",
            "How do I register international travel dates in the mobile app",
            "Heading to {countries} for vacation, will my {card_type} work at foreign terminals",
        ],
        "substitutions": {},
    },
    "card_payment_declined": {
        "domain": "Cards",
        "description": "Find reason for point-of-sale card transaction decline",
        "templates": [
            "Why was my {card_type} declined at {merchants} for {amounts}",
            "My card transaction was rejected even though I have plenty of funds",
            "Declined payment at checkout counter, what is the decline reason code",
            "Payment failed when trying to purchase on {merchants}, please unblock",
        ],
        "substitutions": {},
    },
    "card_unrecognized_charge": {
        "domain": "Cards",
        "description": "Report suspicious or unrecognized charge on card statement",
        "templates": [
            "There is a charge for {amounts} from {merchants} that I never authorized",
            "I see an unrecognized transaction on my card statement from {timeframe}",
            "Suspicious charge on my card statement, I think someone has my card number",
            "Who authorized this {amounts} payment to {merchants}? I did not make it",
        ],
        "substitutions": {},
    },
    "card_cashback_rewards_query": {
        "domain": "Cards",
        "description": "Check or redeem card cash-back and reward points",
        "templates": [
            "How do I redeem my credit card reward points for cash back",
            "What is my current cashback rewards balance this billing cycle",
            "Can I apply my cashback points toward paying down my credit balance",
            "Why didn't I earn 3% cash back on my recent {merchants} purchase",
        ],
        "substitutions": {},
    },
    "card_apple_google_pay_provisioning": {
        "domain": "Cards",
        "description": "Add card to Apple Pay, Google Wallet, or digital wallet",
        "templates": [
            "How do I add my {card_type} to Apple Pay on my {devices}",
            "Having trouble adding my debit card to Google Wallet, verification fails",
            "Can I use Apple Pay immediately while waiting for physical card to arrive",
            "Verification code for Apple Pay digital wallet was not received by SMS",
        ],
        "substitutions": {},
    },

    # ---------------- Domain 2: Transfers, Payments & Remittances (15 Intents) ----------------
    "transfer_domestic_instant": {
        "domain": "Transfers",
        "description": "Initiate or check instant domestic bank-to-bank transfer",
        "templates": [
            "How do I send {amounts} instantly to {relatives}",
            "Send an instant domestic payment to another bank account",
            "Can I transfer {amounts} to {relatives} right now with zero fee",
            "Make an immediate wire transfer to a domestic account number",
        ],
        "substitutions": {},
    },
    "transfer_international_swift": {
        "domain": "Transfers",
        "description": "Wire money overseas via SWIFT or international clearing",
        "templates": [
            "I need to wire {amounts} internationally to {countries}",
            "What is the SWIFT transfer fee and FX exchange rate for sending to {countries}",
            "How long does an international SWIFT wire take to reach a bank in {countries}",
            "Can I transfer funds directly in {currencies} to {relatives} in {countries}",
        ],
        "substitutions": {},
    },
    "transfer_pending_status": {
        "domain": "Transfers",
        "description": "Check why an outgoing or incoming transfer is pending",
        "templates": [
            "Why is my transfer of {amounts} still marked as pending {timeframe}",
            "How long will my bank transfer take to settle in the recipient account",
            "{relatives} sent me {amounts} {timeframe} but my balance hasn't updated",
            "Check current processing status for transfer reference number",
        ],
        "substitutions": {},
    },
    "transfer_failed_reversal": {
        "domain": "Transfers",
        "description": "Failed payment refund and money reversal inquiry",
        "templates": [
            "My transfer of {amounts} failed, when will the money be returned to my balance",
            "Transfer rejected due to invalid account details, how do I get my money back",
            "Payment bounced back, why was the money deducted from my available balance",
            "How many business days for a failed transfer reversal to credit my account",
        ],
        "substitutions": {},
    },
    "transfer_cancel_recall": {
        "domain": "Transfers",
        "description": "Attempt cancellation or recall of a recently sent transfer",
        "templates": [
            "I sent {amounts} to the wrong account number by accident, cancel it immediately",
            "Can I recall a bank transfer that was sent {timeframe}",
            "Stop a pending transfer before it settles in the recipient's bank",
            "Sent money to the wrong person, how do I file an accidental payment recall",
        ],
        "substitutions": {},
    },
    "transfer_add_new_beneficiary": {
        "domain": "Transfers",
        "description": "Save a new payee or contact for wire transfers",
        "templates": [
            "How do I add a new payee with routing number and account number",
            "Steps to save {relatives} as a trusted beneficiary in the mobile app",
            "Add new international payee with IBAN code and recipient address",
            "Why is there a cooling-off security delay when adding a new payee",
        ],
        "substitutions": {},
    },
    "transfer_daily_limit_increase": {
        "domain": "Transfers",
        "description": "Request temporary or permanent wire transfer limit increase",
        "templates": [
            "I need to raise my daily transfer limit to {amounts} for {reasons}",
            "What is the maximum amount I can transfer in a single day",
            "Request a temporary wire limit increase to buy a property",
            "How can I transfer more than {amounts} online without visiting a branch",
        ],
        "substitutions": {},
    },
    "transfer_scheduled_recurring_setup": {
        "domain": "Transfers",
        "description": "Set up automated recurring transfers or standing orders",
        "templates": [
            "How do I schedule a recurring monthly transfer of {amounts} for {reasons}",
            "Set up automatic weekly payment to {relatives} on the 1st of every month",
            "Can I schedule a future-dated transfer to be sent next month",
        ],
        "substitutions": {},
    },
    "transfer_split_bill_request": {
        "domain": "Transfers",
        "description": "Split a transaction or request money from peers",
        "templates": [
            "How do I split a {amounts} dinner bill with my friends in the app",
            "Send a payment request link to {relatives} for their share of the bill",
            "Can I create a group bill split for our vacation expenses",
        ],
        "substitutions": {},
    },
    "transfer_qr_code_pay": {
        "domain": "Transfers",
        "description": "Scan and pay using QR code standard",
        "templates": [
            "How do I scan a merchant QR code to pay with my bank app",
            "Can I display my personal QR code so someone can send me money",
            "QR code payment failed at checkout, how do I retry",
        ],
        "substitutions": {},
    },
    "payment_direct_debit_setup": {
        "domain": "Transfers",
        "description": "Authorize automated direct debit mandate for utilities or bills",
        "templates": [
            "How do I set up a direct debit mandate for my electric utility bill",
            "Where do I find my direct debit authorization form for gym membership",
            "Set up automatic monthly direct debit to pay my insurance premium",
        ],
        "substitutions": {},
    },
    "payment_direct_debit_cancel": {
        "domain": "Transfers",
        "description": "Cancel or revoke active direct debit authorization",
        "templates": [
            "Cancel active direct debit to {merchants} before the next billing date",
            "How do I stop an unauthorized recurring direct debit withdrawal",
            "Remove direct debit mandate from my account permanently",
        ],
        "substitutions": {},
    },
    "payment_standing_order_modify": {
        "domain": "Transfers",
        "description": "Edit or cancel fixed standing order payment",
        "templates": [
            "Modify the amount on my recurring rent standing order to {amounts}",
            "Cancel my standing order payment that goes out on the 15th",
            "How do I change the payment date of an existing standing order",
        ],
        "substitutions": {},
    },
    "payment_utility_bill_pay": {
        "domain": "Transfers",
        "description": "Pay water, electricity, council tax, or internet bills",
        "templates": [
            "I want to pay my water and electricity bill using my checking account",
            "Pay council tax bill directly from the online banking portal",
            "Where is the bill pay feature in the mobile app",
        ],
        "substitutions": {},
    },
    "payment_merchant_refund_inquiry": {
        "domain": "Transfers",
        "description": "Track merchant refund timeline and credit",
        "templates": [
            "{merchants} issued a refund of {amounts} {timeframe}, when will it post",
            "How many business days for an online merchant refund to appear in my balance",
            "Merchant claims they refunded my purchase but I see nothing on statement",
        ],
        "substitutions": {},
    },

    # ---------------- Domain 3: Accounts, Balances & Statements (14 Intents) ----------------
    "account_balance_check": {
        "domain": "Accounts",
        "description": "Check current available balance and cleared funds",
        "templates": [
            "What is my current available balance in checking account",
            "How much money do I have in my savings pot right now",
            "Check my balance including pending authorizations",
            "What is the difference between my current balance and available balance",
        ],
        "substitutions": {},
    },
    "account_statement_download_pdf": {
        "domain": "Accounts",
        "description": "Download official stamped bank statement in PDF format",
        "templates": [
            "How do I download a certified PDF bank statement for the last 3 months",
            "Need an official stamped statement for mortgage application",
            "Export my transaction history for tax filing in CSV or PDF format",
            "Where can I find electronic monthly statements in online banking",
        ],
        "substitutions": {},
    },
    "account_overdraft_apply": {
        "domain": "Accounts",
        "description": "Apply for arranged overdraft facility or increase limit",
        "templates": [
            "Can I apply for an arranged overdraft limit of {amounts}",
            "How do I increase my existing overdraft buffer on checking account",
            "Check my eligibility for an interest-free overdraft facility",
        ],
        "substitutions": {},
    },
    "account_overdraft_fee_inquiry": {
        "domain": "Accounts",
        "description": "Overdraft charges and interest fee inquiries",
        "templates": [
            "What is the daily interest rate if I go into unarranged overdraft",
            "Why was I charged an overdraft fee of {amounts} {timeframe}",
            "Can you waive the overdraft charge since my salary was deposited today",
        ],
        "substitutions": {},
    },
    "account_interest_rate_query": {
        "domain": "Accounts",
        "description": "Savings account interest rates, APY, and calculation",
        "templates": [
            "What is the current annual APY interest rate on high-yield savings",
            "How often is interest calculated and paid on my savings account",
            "Did interest rates change on fixed-term deposit accounts recently",
        ],
        "substitutions": {},
    },
    "account_closure_request": {
        "domain": "Accounts",
        "description": "Close checking or savings account permanently",
        "templates": [
            "I want to permanently close my checking account",
            "What is the procedure to shut down my account and transfer remaining funds",
            "How do I close my bank account with zero remaining balance",
            "Please terminate my account and send confirmation of closure",
        ],
        "substitutions": {},
    },
    "account_switch_service_cassy": {
        "domain": "Accounts",
        "description": "Current account switch service from competitor bank",
        "templates": [
            "How does the automated current account switch guarantee work",
            "Will all my direct debits transfer automatically if I switch banks",
            "What is the status of my bank switch from my old account",
        ],
        "substitutions": {},
    },
    "account_joint_holder_add": {
        "domain": "Accounts",
        "description": "Add spouse or partner as joint account holder",
        "templates": [
            "How can I add my spouse as a joint owner on my checking account",
            "Can two people manage the same bank account with separate debit cards",
            "Requirements and identity documents to convert sole account to joint",
        ],
        "substitutions": {},
    },
    "account_dormant_reactivation": {
        "domain": "Accounts",
        "description": "Reactivate inactive or dormant bank account",
        "templates": [
            "My account was marked dormant because of inactivity, how to reopen it",
            "Can I reactivate an old savings account that has been locked",
            "What documents are needed to reclaim funds from a dormant account",
        ],
        "substitutions": {},
    },
    "account_change_residential_address": {
        "domain": "Accounts",
        "description": "Update home address on bank records",
        "templates": [
            "I moved to a new house, how do I update my registered address",
            "Change my billing and mailing address in account settings",
            "Do I need to provide utility bill proof to change address online",
        ],
        "substitutions": {},
    },
    "account_change_phone_email": {
        "domain": "Accounts",
        "description": "Update registered telephone number or email address",
        "templates": [
            "I have a new mobile phone number, update it for 2FA SMS alerts",
            "Change the primary email address linked to my online banking",
            "Why am I not receiving OTP codes after updating my phone number",
        ],
        "substitutions": {},
    },
    "account_open_savings_pot": {
        "domain": "Accounts",
        "description": "Create savings sub-account, goal pot, or vault",
        "templates": [
            "How do I create a new savings pot with round-up spare change feature",
            "Can I lock a savings goal pot until a specific target date",
            "Open an additional high-interest savings sub-account in the app",
        ],
        "substitutions": {},
    },
    "account_iban_bic_swift_lookup": {
        "domain": "Accounts",
        "description": "Find IBAN, BIC, SWIFT, and sort code / routing number",
        "templates": [
            "Where do I find my account IBAN and SWIFT BIC code in the app",
            "What is the bank routing transit number for direct wire deposits",
            "I need my international bank account number to receive money from {countries}",
        ],
        "substitutions": {},
    },
    "account_tax_certificate_request": {
        "domain": "Accounts",
        "description": "Request annual tax interest certificate or 1099-INT form",
        "templates": [
            "Where can I download my annual tax statement and 1099-INT form",
            "Request certificate of interest earned for tax filing purposes",
            "Download annual interest statement for the tax year",
        ],
        "substitutions": {},
    },

    # ---------------- Domain 4: Cash, ATMs & Physical Banking (12 Intents) ----------------
    "atm_cash_withdrawal_declined": {
        "domain": "Cash & ATM",
        "description": "ATM declined cash withdrawal transaction",
        "templates": [
            "ATM refused to dispense cash for {amounts} even though I have balance",
            "Why did the cash machine decline my withdrawal attempt {timeframe}",
            "ATM error screen when trying to withdraw cash, what went wrong",
        ],
        "substitutions": {},
    },
    "atm_incorrect_amount_dispensed": {
        "domain": "Cash & ATM",
        "description": "ATM gave wrong amount of cash (short dispensed or over dispensed)",
        "templates": [
            "I requested {amounts} at the ATM but the machine only gave me half",
            "ATM charged my account for {amounts} but dispensed no cash at all",
            "Cash machine gave wrong banknote count, how do I file a discrepancy claim",
        ],
        "substitutions": {},
    },
    "atm_deposit_cash_not_credited": {
        "domain": "Cash & ATM",
        "description": "Cash deposit at ATM not credited to balance",
        "templates": [
            "I deposited {amounts} in cash at the ATM machine but my balance didn't update",
            "Cash deposit machine jammed while counting my banknotes",
            "Deposited cash at smart ATM {timeframe}, where is the money",
        ],
        "substitutions": {},
    },
    "atm_locator_nearest_branch": {
        "domain": "Cash & ATM",
        "description": "Find nearest physical branch or free cash withdrawal ATM",
        "templates": [
            "Where is the nearest surcharge-free ATM in my area",
            "Locate closest open physical branch with counter service",
            "Find cash deposit-capable ATMs near my current location",
        ],
        "substitutions": {},
    },
    "check_deposit_mobile_camera": {
        "domain": "Cash & ATM",
        "description": "Deposit paper check using mobile app camera capture",
        "templates": [
            "How do I deposit a paper check using mobile check deposit feature",
            "Mobile check deposit failed to read the endorsement signature on back",
            "What is the maximum limit for mobile camera check deposits",
        ],
        "substitutions": {},
    },
    "check_order_checkbook": {
        "domain": "Cash & ATM",
        "description": "Order new paper checkbook",
        "templates": [
            "I ran out of paper checks, how do I order a new checkbook",
            "Order standard 25-check book for checking account",
            "How long does it take for a new checkbook to arrive by post",
        ],
        "substitutions": {},
    },
    "check_stop_payment_order": {
        "domain": "Cash & ATM",
        "description": "Issue stop payment request on a written check",
        "templates": [
            "I need to place an urgent stop payment order on check number 1042",
            "Stop check written for {amounts} before it gets cashed",
            "What is the fee to cancel and stop payment on a written check",
        ],
        "substitutions": {},
    },
    "check_clearance_hold_inquiry": {
        "domain": "Cash & ATM",
        "description": "Inquire about check funds availability and clearing holds",
        "templates": [
            "Why is there a 5-day funds hold on my deposited check of {amounts}",
            "When will funds from my deposited check become fully available for spending",
            "Can you release the clearance hold on my payroll check early",
        ],
        "substitutions": {},
    },
    "cash_foreign_currency_order": {
        "domain": "Cash & ATM",
        "description": "Order foreign currency banknotes for international travel",
        "templates": [
            "Can I order foreign currency cash in {currencies} before traveling to {countries}",
            "Order {currencies} cash banknotes for collection at local branch",
            "What is the foreign exchange commission rate for travel cash",
        ],
        "substitutions": {},
    },
    "cash_deposit_counter_hours": {
        "domain": "Cash & ATM",
        "description": "Check counter cash handling hours and guidelines",
        "templates": [
            "What are the branch counter opening hours for large cash deposits",
            "Can I deposit coins and loose cash at the teller counter",
            "Do I need to make an appointment to deposit more than {amounts} in cash",
        ],
        "substitutions": {},
    },
    "atm_withdrawal_limit_query": {
        "domain": "Cash & ATM",
        "description": "Daily ATM cash withdrawal limit inquiries",
        "templates": [
            "What is the maximum daily ATM cash withdrawal limit on my debit card",
            "Can I temporarily increase my daily ATM cash withdrawal limit to {amounts}",
            "ATM says I reached my daily cash limit, when does the 24-hour limit reset",
        ],
        "substitutions": {},
    },
    "atm_operator_fee_refund": {
        "domain": "Cash & ATM",
        "description": "Refund inquiry for out-of-network ATM surcharge fees",
        "templates": [
            "An out-of-network ATM charged me a $3.50 fee, do you reimburse ATM fees",
            "ATM surcharge fee refund inquiry on checking account",
            "Does my account type include worldwide free ATM withdrawals",
        ],
        "substitutions": {},
    },

    # ---------------- Domain 5: Loans, Mortgages & Financing (14 Intents) ----------------
    "loan_personal_application": {
        "domain": "Loans",
        "description": "Apply for an unsecured personal loan",
        "templates": [
            "How do I apply for an unsecured personal loan of {amounts}",
            "What are the interest rates and terms for a personal loan for {reasons}",
            "Check my pre-approved personal loan offers in the app",
        ],
        "substitutions": {},
    },
    "loan_eligibility_calculator": {
        "domain": "Loans",
        "description": "Calculate loan borrowing capacity and monthly installments",
        "templates": [
            "What would my monthly repayment be on a {amounts} loan over 3 years",
            "Calculate borrowing power and APR for personal financing",
            "Check loan affordability without impacting my credit score",
        ],
        "substitutions": {},
    },
    "loan_early_payoff_quotation": {
        "domain": "Loans",
        "description": "Request early loan settlement quotation",
        "templates": [
            "How much is the total early settlement figure to pay off my loan today",
            "Is there an early repayment penalty fee if I clear my loan balance early",
            "Generate early payoff quotation statement for my personal loan",
        ],
        "substitutions": {},
    },
    "loan_interest_rate_reduction": {
        "domain": "Loans",
        "description": "Inquire about lowering APR interest rate on existing loan",
        "templates": [
            "Can I lower the APR interest rate on my existing personal loan",
            "Negotiate better interest rate on borrowing based on improved credit score",
            "Refinance my existing loan to get a cheaper monthly payment",
        ],
        "substitutions": {},
    },
    "loan_payment_date_change": {
        "domain": "Loans",
        "description": "Change monthly loan installment direct debit due date",
        "templates": [
            "Can I change my monthly loan payment due date to match my payday",
            "Move loan repayment deduction date from 1st to 25th of the month",
            "How do I change the payment date on my existing installment plan",
        ],
        "substitutions": {},
    },
    "loan_payment_holiday_deferral": {
        "domain": "Loans",
        "description": "Request temporary payment pause or loan holiday",
        "templates": [
            "Can I request a 2-month loan repayment holiday due to financial hardship",
            "Defer loan payment this month due to unexpected {reasons}",
            "How does applying for a loan payment pause affect my total interest",
        ],
        "substitutions": {},
    },
    "mortgage_application_status": {
        "domain": "Loans",
        "description": "Track home mortgage loan underwriting status",
        "templates": [
            "What is the current underwriting status of my mortgage application",
            "Has the property valuation appraisal been completed for my mortgage",
            "How long until my mortgage offer letter is issued by the underwriters",
        ],
        "substitutions": {},
    },
    "mortgage_remortgage_options": {
        "domain": "Loans",
        "description": "Remortgage or switch mortgage deal upon rate expiry",
        "templates": [
            "My fixed mortgage deal expires in 3 months, what are my remortgage options",
            "Switch to a new fixed-rate mortgage deal to avoid standard variable rate",
            "What fees are involved in remortgaging to an alternative product",
        ],
        "substitutions": {},
    },
    "mortgage_overpayment_allowance": {
        "domain": "Loans",
        "description": "Overpay mortgage balance without penalty",
        "templates": [
            "How much can I overpay on my mortgage each year without early exit charges",
            "Make a lump sum overpayment of {amounts} against mortgage principal",
            "Will overpaying reduce my monthly payment or shorten mortgage term",
        ],
        "substitutions": {},
    },
    "mortgage_fixed_rate_expiry": {
        "domain": "Loans",
        "description": "Inquiries regarding expiration of promotional mortgage fixed rate",
        "templates": [
            "When exactly does my 5-year fixed mortgage promotional rate end",
            "What will my monthly mortgage payment increase to on the standard rate",
            "How early can I lock in a new mortgage rate before fixed term ends",
        ],
        "substitutions": {},
    },
    "credit_limit_increase_request": {
        "domain": "Loans",
        "description": "Request credit line limit increase on credit card",
        "templates": [
            "I want to request an increase to my credit card spending limit",
            "Can I raise my credit card credit line from {amounts} to a higher limit",
            "Am I eligible for an automatic credit limit increase on my account",
        ],
        "substitutions": {},
    },
    "credit_score_impact_inquiry": {
        "domain": "Loans",
        "description": "Understand credit score checks and inquiries",
        "templates": [
            "Does applying for an overdraft or loan perform a hard credit search",
            "Why did my credit score drop after opening a new bank account",
            "How do I check my credit report and score inside the mobile app",
        ],
        "substitutions": {},
    },
    "debt_consolidation_query": {
        "domain": "Loans",
        "description": "Consolidate multiple credit card and loan debts into one payment",
        "templates": [
            "Can I consolidate all my credit card debts into a single low-interest loan",
            "How does debt consolidation work with your personal loan product",
            "Combine multiple monthly debts into one simple monthly payment",
        ],
        "substitutions": {},
    },
    "student_loan_refinancing": {
        "domain": "Loans",
        "description": "Refinance or manage student education financing loans",
        "templates": [
            "Can I refinance my private student loan to get a lower interest rate",
            "What are the terms and interest rates for student loan refinancing",
            "Check my eligibility to consolidate my education debt",
        ],
        "substitutions": {},
    },

    # ---------------- Domain 6: Investments, Wealth & Crypto (12 Intents) ----------------
    "invest_portfolio_performance_summary": {
        "domain": "Investments",
        "description": "View total portfolio return, gains, and performance summary",
        "templates": [
            "What is my total investment portfolio return year-to-date",
            "Show summary of gains and losses across my investment stocks",
            "How is my robo-advisor diversified investment fund performing",
        ],
        "substitutions": {},
    },
    "invest_buy_sell_etf_stock": {
        "domain": "Investments",
        "description": "Execute stock, index fund, or ETF trading orders",
        "templates": [
            "How do I buy shares of index funds or ETFs in the investment hub",
            "Place a market order to sell stocks in my investment portfolio",
            "What are the trading commission fees per stock purchase",
        ],
        "substitutions": {},
    },
    "invest_dividend_reinvestment_plan": {
        "domain": "Investments",
        "description": "Enable automated dividend reinvestment (DRIP)",
        "templates": [
            "How do I enable automatic dividend reinvestment DRIP on my stocks",
            "When will cash dividends from my index funds be paid into my balance",
            "Can I choose to payout dividends in cash instead of reinvesting",
        ],
        "substitutions": {},
    },
    "invest_isa_allowance_limit": {
        "domain": "Investments",
        "description": "Tax-free ISA or Roth IRA annual contribution limits",
        "templates": [
            "How much remaining contribution allowance do I have in my Stocks & Shares ISA",
            "What is the maximum annual tax-free investment contribution limit",
            "Can I transfer my Cash ISA to a Stocks and Shares ISA without losing allowance",
        ],
        "substitutions": {},
    },
    "invest_robo_advisor_risk_profile": {
        "domain": "Investments",
        "description": "Adjust automated robo-advisor risk tolerance profile",
        "templates": [
            "Change my investment risk profile from aggressive growth to conservative",
            "How do I retake the risk tolerance questionnaire for my managed portfolio",
            "Adjust asset allocation weighting between equities and bonds",
        ],
        "substitutions": {},
    },
    "invest_capital_gains_tax_report": {
        "domain": "Investments",
        "description": "Download investment capital gains and tax reporting statement",
        "templates": [
            "Where do I download my annual capital gains tax report for investments",
            "Need tax statement showing realized gains and dividend income for IRS filing",
            "Export annual investment tax statement in PDF format",
        ],
        "substitutions": {},
    },
    "invest_mutual_fund_fees": {
        "domain": "Investments",
        "description": "Ongoing expense ratios and platform management fee query",
        "templates": [
            "What are the annual platform fees for holding investment funds",
            "Check ongoing charges figure OCF on my managed mutual funds",
            "How are account custody fees deducted from my investment balance",
        ],
        "substitutions": {},
    },
    "crypto_wallet_withdrawal": {
        "domain": "Investments",
        "description": "Transfer cryptocurrency to external wallet address",
        "templates": [
            "How do I withdraw Bitcoin or Ethereum to an external crypto wallet address",
            "Why is my crypto withdrawal pending blockchain network confirmation",
            "What is the daily withdrawal limit for cryptocurrency assets",
        ],
        "substitutions": {},
    },
    "crypto_purchase_supported_coins": {
        "domain": "Investments",
        "description": "Buy cryptocurrency tokens and check supported coins",
        "templates": [
            "How do I buy cryptocurrency directly using my fiat bank balance",
            "Which crypto tokens are supported for trading in the banking app",
            "Can I set up recurring automatic purchases of Bitcoin",
        ],
        "substitutions": {},
    },
    "crypto_network_gas_fee_query": {
        "domain": "Investments",
        "description": "Blockchain network gas and transaction fee inquiries",
        "templates": [
            "What network fee is charged when sending crypto transactions",
            "Why are Ethereum gas fees so high on this crypto transfer",
            "Are there hidden conversion markups on crypto purchases",
        ],
        "substitutions": {},
    },
    "pension_retirement_transfer": {
        "domain": "Investments",
        "description": "Consolidate or transfer workplace pension plans",
        "templates": [
            "How do I consolidate old workplace pensions into my self-invested pension SIPP",
            "Combine pension pots from previous employers into one managed fund",
            "What are the tax relief benefits on personal pension contributions",
        ],
        "substitutions": {},
    },
    "wealth_private_banking_adviser": {
        "domain": "Investments",
        "description": "Request consultation with private wealth manager",
        "templates": [
            "Can I speak with a dedicated private wealth manager regarding my assets",
            "What are the qualification thresholds for private banking services",
            "Book an estate planning and inheritance tax consultation",
        ],
        "substitutions": {},
    },

    # ---------------- Domain 7: Security, Authentication & Fraud (14 Intents) ----------------
    "security_unauthorized_transaction_alert": {
        "domain": "Security",
        "description": "Report fraudulent activity and compromised account access",
        "templates": [
            "I received an alert about a login from an unknown location, help",
            "Money was moved out of my account without my permission, fraud alert",
            "Someone has gained unauthorized access to my online banking, freeze everything",
            "Emergency: my account credentials have been compromised, lock access",
        ],
        "substitutions": {},
    },
    "security_identity_verification_kyc": {
        "domain": "Security",
        "description": "Submit passport or ID documents for KYC verification",
        "templates": [
            "How do I complete identity verification KYC in the mobile app",
            "Uploaded my passport {timeframe}, why is identity verification pending",
            "What photo identification documents are accepted for identity proof",
            "App says I need to re-verify my identity to unblock transfer features",
        ],
        "substitutions": {},
    },
    "security_two_factor_auth_reset": {
        "domain": "Security",
        "description": "Reset 2FA authenticator app or phone verification",
        "templates": [
            "I got a new phone and lost access to my 2FA authenticator app",
            "Reset two-factor authentication so I can log in from my new {devices}",
            "Not receiving two-factor SMS security codes to my mobile number",
        ],
        "substitutions": {},
    },
    "security_password_forgotten": {
        "domain": "Security",
        "description": "Forgot login password or unlock online banking access",
        "templates": [
            "I forgot my online banking password, how do I reset it securely",
            "Account locked out after entering incorrect password multiple times",
            "Send a secure password reset link to my registered email address",
        ],
        "substitutions": {},
    },
    "security_biometric_faceid_touchid_error": {
        "domain": "Security",
        "description": "Fix FaceID, fingerprint, or biometric login errors",
        "templates": [
            "FaceID biometric login is failing on my {devices}, how to re-enable",
            "Fingerprint authentication not recognized when opening the banking app",
            "Can I use biometric authentication instead of typing my password",
        ],
        "substitutions": {},
    },
    "security_suspicious_phishing_email_report": {
        "domain": "Security",
        "description": "Forward or report phishing email, SMS smishing, or scam text",
        "templates": [
            "I received a suspicious text message asking for my bank login details",
            "How do I report a phishing email claiming to be from your fraud team",
            "Is this text message about a frozen account legitimate or a scam",
        ],
        "substitutions": {},
    },
    "security_unknown_device_login_warning": {
        "domain": "Security",
        "description": "Inquire about unrecognized device login notification",
        "templates": [
            "Got an alert that an unknown device logged into my account in {countries}",
            "Why did I get a new device registration alert if I didn't login",
            "Revoke trusted device status for an old {devices} I sold",
        ],
        "substitutions": {},
    },
    "security_scam_call_verification": {
        "domain": "Security",
        "description": "Verify if a telephone call was genuine bank staff or scammer",
        "templates": [
            "Someone called me claiming to be from your fraud department, was it real",
            "A caller asked me to move my money to a safe account, is this a scam",
            "Verify whether telephone number 800-123-4567 is an official bank number",
        ],
        "substitutions": {},
    },
    "security_trusted_beneficiary_cooling_period": {
        "domain": "Security",
        "description": "Inquire about 24-hour security delay for new payee transfers",
        "templates": [
            "Why is my payment to a new payee held for a 24-hour security delay",
            "Can you expedite the fraud cooling-off period on my new payee transfer",
            "How long is the mandatory waiting period after adding a new payee",
        ],
        "substitutions": {},
    },
    "security_passport_id_document_rejected": {
        "domain": "Security",
        "description": "Assistance when uploaded ID document is rejected by system",
        "templates": [
            "My uploaded passport was rejected due to glare, how do I re-submit",
            "Driver's license photo verification failed, what should I do",
            "Why did the automated document scanner reject my national ID card",
        ],
        "substitutions": {},
    },
    "security_freeze_all_digital_access": {
        "domain": "Security",
        "description": "Emergency lockdown of all online banking, cards, and logins",
        "templates": [
            "Emergency lockdown: freeze all digital logins, cards, and transactions immediately",
            "I was scammed, please lock all banking channels right now",
            "Place a total security freeze on my entire bank profile",
        ],
        "substitutions": {},
    },
    "security_update_security_questions": {
        "domain": "Security",
        "description": "Update account security challenge questions and answers",
        "templates": [
            "How do I update my memorable security questions and answers",
            "Forgot the answers to my security questions, how do I reset them",
            "Change my secret authentication questions in profile settings",
        ],
        "substitutions": {},
    },
    "security_data_breach_inquiry": {
        "domain": "Security",
        "description": "Inquire if personal customer data was impacted in security leaks",
        "templates": [
            "Was my account or card information impacted by recent cybersecurity news",
            "How does your bank protect customer data against credential stuffing breaches",
            "Has my personal account information ever been exposed",
        ],
        "substitutions": {},
    },
    "security_device_lost_revoke_tokens": {
        "domain": "Security",
        "description": "De-register lost phone or revoke digital session tokens",
        "templates": [
            "I lost my {devices}, please log out all active app sessions immediately",
            "De-register my stolen mobile phone from trusted devices list",
            "Revoke all active biometric access tokens across all devices",
        ],
        "substitutions": {},
    },

    # ---------------- Domain 8: Fees, Disputes & Customer Support (13 Intents) ----------------
    "fee_monthly_maintenance_waiver": {
        "domain": "Fees & Support",
        "description": "Inquire about monthly account maintenance fee or waiver criteria",
        "templates": [
            "Why was I charged a monthly maintenance fee of {amounts}",
            "How do I qualify for fee-free checking and waive monthly account fees",
            "Can you refund the monthly service fee charged to my balance {timeframe}",
        ],
        "substitutions": {},
    },
    "fee_international_transaction_charge": {
        "domain": "Fees & Support",
        "description": "Foreign transaction FX markup fee dispute",
        "templates": [
            "Why was there a 3% foreign transaction fee on my {merchants} purchase",
            "Do you charge currency conversion fees when spending abroad in {currencies}",
            "Can you reverse the non-sterling transaction fee on my card",
        ],
        "substitutions": {},
    },
    "fee_wire_transfer_cost_query": {
        "domain": "Fees & Support",
        "description": "Domestic and international wire transfer service fee query",
        "templates": [
            "How much does it cost to send an outgoing domestic wire transfer",
            "What is the total fee for sending an international wire in {currencies}",
            "Are incoming wire transfers free of charge on this checking account",
        ],
        "substitutions": {},
    },
    "fee_chargeback_status_update": {
        "domain": "Fees & Support",
        "description": "Track open dispute or merchant chargeback claim",
        "templates": [
            "What is the status of my chargeback dispute against {merchants}",
            "When will provisional credit for my disputed charge become permanent",
            "Upload supporting evidence documents for my ongoing merchant dispute",
        ],
        "substitutions": {},
    },
    "fee_late_payment_penalty_dispute": {
        "domain": "Fees & Support",
        "description": "Request waiver or dispute late credit card payment fee",
        "templates": [
            "I missed my credit card payment due date by one day, please waive the late fee",
            "Dispute late payment penalty fee charged {timeframe}",
            "Will a one-time late payment penalty be reported to credit bureaus",
        ],
        "substitutions": {},
    },
    "complaint_poor_service_escalation": {
        "domain": "Fees & Support",
        "description": "File formal complaint regarding customer service experience",
        "templates": [
            "I want to file a formal complaint about poor customer service at local branch",
            "Escalate my issue to a senior operations supervisor immediately",
            "How do I submit an official customer grievance to executive complaints team",
        ],
        "substitutions": {},
    },
    "complaint_ombudsman_referral": {
        "domain": "Fees & Support",
        "description": "Request final response letter for Financial Ombudsman escalation",
        "templates": [
            "Send me your final response letter so I can escalate to the Financial Ombudsman",
            "How do I refer an unresolved banking dispute to the Ombudsman service",
            "I reject your complaint outcome and am contacting financial regulators",
        ],
        "substitutions": {},
    },
    "support_talk_to_human_agent": {
        "domain": "Fees & Support",
        "description": "Request live human representative or specialist transfer",
        "templates": [
            "Transfer me to a live human customer support agent right now",
            "I want to speak with a real person, not an automated bot",
            "Connect me with a live representative to handle a complex issue",
        ],
        "substitutions": {},
    },
    "support_schedule_branch_appointment": {
        "domain": "Fees & Support",
        "description": "Book appointment with specialist at physical branch",
        "templates": [
            "How do I book an in-person appointment with a mortgage adviser at local branch",
            "Schedule a face-to-face meeting at the branch for next week",
            "Cancel or reschedule my existing branch appointment",
        ],
        "substitutions": {},
    },
    "support_operating_hours_holidays": {
        "domain": "Fees & Support",
        "description": "Customer service telephone and branch operating hours",
        "templates": [
            "What are your customer service telephone support hours on weekends",
            "Is the branch open on bank holidays or Sundays",
            "What hours is the 24/7 fraud hotline staffed",
        ],
        "substitutions": {},
    },
    "gdpr_right_to_be_forgotten": {
        "domain": "Fees & Support",
        "description": "Request personal data deletion under GDPR / CCPA privacy law",
        "templates": [
            "I want to exercise my GDPR right to be forgotten and erase personal data",
            "Delete my personal marketing profile and records under CCPA privacy rules",
            "How long do you retain financial records after account closure",
        ],
        "substitutions": {},
    },
    "gdpr_subject_access_request_export": {
        "domain": "Fees & Support",
        "description": "Request full GDPR Subject Access Request (SAR) data export",
        "templates": [
            "I want to submit a formal GDPR Subject Access Request for all my data",
            "Export all customer communication transcripts and profile records",
            "How do I receive a complete digital copy of all data your bank holds on me",
        ],
        "substitutions": {},
    },
    "accessibility_audio_braille_services": {
        "domain": "Fees & Support",
        "description": "Request accessible banking materials (Braille, large print, audio)",
        "templates": [
            "Can I receive monthly bank statements in Braille or large print",
            "How do I enable high-contrast screen reader accessibility in the app",
            "Request sign language BSL interpreter services for branch appointment",
        ],
        "substitutions": {},
    },
}

# --------------------------------------------------------------------------
# 3. High-Speed Vectorized Combinatorial Generator
# --------------------------------------------------------------------------
PREFIXES = [
    "", "Hello, ", "Hi there, ", "Excuse me, ", "Urgent: ", "Question: ",
    "Could you tell me, ", "I need help: ", "Please explain, ", "Good morning, "
]

SUFFIXES = [
    "", " please.", " as soon as possible.", " right now.", " thank you.",
    " thanks for your help.", " it's very urgent.", " appreciate it.", " ASAP."
]


SLOT_ALIASES = {
    "card_type": "card_types",
    "card_types": "card_types",
    "amount": "amounts",
    "amounts": "amounts",
    "merchant": "merchants",
    "merchants": "merchants",
    "timeframe": "timeframes",
    "timeframes": "timeframes",
    "relative": "relatives",
    "relatives": "relatives",
    "country": "countries",
    "countries": "countries",
    "currency": "currencies",
    "currencies": "currencies",
    "device": "devices",
    "devices": "devices",
    "reason": "reasons",
    "reasons": "reasons",
}

import re

STARTERS = [
    "How do I", "Can I", "I need to", "Please help me", "I want to", "Where can I",
    "Is it possible to", "I am trying to", "What are the steps to", "Kindly assist me to",
    "Could you explain how to", "I would like to", "Why can't I", "Who do I contact to",
    "Tell me how to", "Can someone help me", "Please advise on how to", "I must",
    "Do you know how to", "Show me how to", "Can your system", "I am requesting to",
]

MODIFIERS = [
    "urgently", "immediately", "right now", "today", "online", "via the mobile app",
    "as soon as possible", "promptly", "safely", "without delay", "this morning",
    "at the earliest", "from my account", "for my profile", "through internet banking",
    "on my device", "before the end of the day", "straight away", "ASAP"
]

REASONS = [
    "for an emergency", "for security reasons", "because of a mistake", "due to an error",
    "for my peace of mind", "to avoid extra fees", "before it gets worse", "right away",
    "due to an unexpected issue", "since I am traveling", "for verification", "on my statement"
]


def generate_intent_utterances(intent_key: str, spec: dict, target_count: int = 2000) -> list[str]:
    """
    Generates target_count unique, realistic customer utterances for a single intent.
    Combines template expansions, slot fillers, prefixes, suffixes, and linguistic perturbations.
    """
    templates = spec["templates"]
    subs = spec.get("substitutions", {})
    generated_set = set()

    # Pre-generate combinatorial pool from spec templates
    for template in templates:
        placeholders = list(set(re.findall(r"\{(\w+)\}", template)))

        slot_options = {}
        for p in placeholders:
            if p in subs:
                slot_options[p] = subs[p]
            elif p in SLOT_ALIASES and SLOT_ALIASES[p] in SLOTS:
                slot_options[p] = SLOTS[SLOT_ALIASES[p]]
            elif p in SLOTS:
                slot_options[p] = SLOTS[p]
            else:
                slot_options[p] = ["item"]

        if slot_options:
            keys = list(slot_options.keys())
            values_lists = [slot_options[k] for k in keys]
            combinations = list(itertools.product(*values_lists))
            random.shuffle(combinations)

            for combo in combinations[:500]:
                filler = dict(zip(keys, combo))
                utterance = template.format(**filler)
                generated_set.add(utterance)
        else:
            generated_set.add(template)

    base_list = list(generated_set)

    # Use combinatorial cross-product of starters, modifiers, and reasons to reach target_count
    for base in base_list:
        if len(generated_set) >= target_count:
            break
        # Strip trailing punctuation
        clean_base = base.rstrip(".?! ")
        clean_base_lower = clean_base[0].lower() + clean_base[1:] if clean_base else clean_base

        for s in STARTERS:
            for m in MODIFIERS:
                generated_set.add(f"{s} {clean_base_lower} {m}?")
                generated_set.add(f"{s} {clean_base_lower}?")
                generated_set.add(f"{clean_base} {m}.")
                if len(generated_set) >= target_count:
                    break
            if len(generated_set) >= target_count:
                break

    # If still needing more, add ticket / reference numbers and date variations
    counter = 1000
    while len(generated_set) < target_count:
        base = random.choice(base_list)
        clean_base = base.rstrip(".?! ")
        s = random.choice(STARTERS)
        m = random.choice(MODIFIERS)
        r = random.choice(REASONS)
        variant = f"{s} {clean_base[0].lower() + clean_base[1:]} {m} {r} (ref #{counter})."
        generated_set.add(variant)
        counter += 1

    return list(generated_set)[:target_count]


def build_banking_200k_dataset(samples_per_intent: int = 2000) -> tuple[pd.DataFrame, dict]:
    """
    Generates 220,000 utterances across 110 intents.
    Returns (DataFrame, taxonomy_dict).
    """
    print(f"[Dataset Generator] Generating {len(INTENT_SPECS)} intents x {samples_per_intent} samples = {len(INTENT_SPECS) * samples_per_intent:,} total rows...")

    all_rows = []
    taxonomy = {}

    for idx, (intent, spec) in enumerate(INTENT_SPECS.items(), start=1):
        utterances = generate_intent_utterances(intent, spec, target_count=samples_per_intent)
        for text in utterances:
            all_rows.append({
                "text": text,
                "label_text": intent,
                "domain": spec["domain"],
            })
        taxonomy[intent] = {
            "intent_id": idx - 1,
            "domain": spec["domain"],
            "description": spec["description"],
            "sample_count": len(utterances),
        }

    random.shuffle(all_rows)
    df = pd.DataFrame(all_rows)

    # Encode integer labels
    unique_labels = sorted(list(set(df["label_text"])))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    df["label"] = df["label_text"].map(label2id)

    print(f"[Dataset Generator] Successfully generated {len(df):,} rows across {len(unique_labels)} unique intents!")
    return df, taxonomy


def export_dataset(df: pd.DataFrame, taxonomy: dict):
    """Saves the dataset to Parquet and taxonomy to JSON."""
    parquet_path = DATA_DIR / "banking_intent_200k.parquet"
    taxonomy_path = DATA_DIR / "intent_taxonomy_110.json"
    summary_path = DATA_DIR / "banking_intent_200k_summary.json"

    print(f"[Exporter] Saving to Parquet: {parquet_path}...")
    df.to_parquet(parquet_path, index=False, compression="snappy")

    print(f"[Exporter] Saving Taxonomy JSON: {taxonomy_path}...")
    with open(taxonomy_path, "w") as f:
        json.dump(taxonomy, f, indent=2)

    domain_counts = df["domain"].value_counts().to_dict()
    summary = {
        "total_rows": len(df),
        "total_intents": len(taxonomy),
        "domain_distribution": domain_counts,
        "sample_rows": df.head(5).to_dict(orient="records"),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    file_size_mb = os.path.getsize(parquet_path) / (1024 ** 2)
    print(f"[Exporter] Complete! Dataset file size: {file_size_mb:.2f} MB on disk.")


if __name__ == "__main__":
    df, taxonomy = build_banking_200k_dataset(samples_per_intent=2000)
    export_dataset(df, taxonomy)
