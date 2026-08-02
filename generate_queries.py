import ollama

merchq = str(input("enter the merchant query: "))
for i in range(5):
    response = ollama.chat(
        model='llama3.2',
        messages=[
            {'role': 'system', 'content': """you are a shopify graphql expert. generate only valid shopify admin graphql queries using real fields.

customer fields: id, email, firstName, lastName, customer_account_status (values: Enabled, Disabled, Declined, Invited), tags, amountSpent, numberOfOrders, createdAt

order fields: id, name, createdAt, totalPriceSet, displayFinancialStatus, displayFulfillmentStatus, cancelledAt, tags, customer

use edges/nodes pagination pattern. return only the raw graphql query, no explanation, no backticks."""},
            {'role': 'user', 'content': merchq}
        ],
        options={'temperature': 0.78}
    )
    print(f"\n--- query {i+1} ---")
    print(response.message.content)