import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """you are a shopify graphql expert. given a merchant question, generate only a valid shopify admin graphql query. return only the raw query, no explanation, no markdown, no backticks.

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


def generate_queries(merchant_question: str, n: int = 5) -> list[str]:
    """generate n candidate graphql queries for a merchant question at temperature 0.8"""
    queries = []
    for _ in range(n):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            temperature=0.8,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": merchant_question}]
        )
        queries.append(response.content[0].text)
    return queries


if __name__ == "__main__":
    merchant_question = input("enter the merchant query: ")
    for i, q in enumerate(generate_queries(merchant_question), start=1):
        print(f"\n--- query {i} ---")
        print(q)
