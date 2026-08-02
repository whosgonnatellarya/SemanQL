import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

system_prompt = """you are a shopify graphql expert. given a merchant question, generate only a valid shopify admin graphql query. return only the raw query, no explanation, no markdown, no backticks.

use only these real shopify admin api fields:

customers query fields:
- id, email, firstName, lastName
- amountSpent (for filtering by spend)
- numberOfOrders
- createdAt
- tags (array of strings)
- customer_account_status (values: Enabled, Disabled, Declined, Invited)

orders query fields:
- id, name, createdAt
- totalPriceSet
- displayFinancialStatus
- displayFulfillmentStatus
- cancelledAt
- tags
- customer

filtering customers:
- use query argument with string syntax like: query: "state:enabled AND amount_spent:>500"
- valid filter keys: state, amount_spent, email, tag, country

pagination:
- use first: and after: arguments
- use edges { node { } } or nodes { } pattern"""

merchq = str(input("enter the merchant query: "))

for i in range(5):
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        temperature=0.8,
        system=system_prompt,
        messages=[{"role": "user", "content": merchq}]
    )
    print(f"\n--- query {i+1} ---")
    print(response.content[0].text)