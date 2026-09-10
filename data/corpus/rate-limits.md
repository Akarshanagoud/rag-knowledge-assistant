# API Rate Limits

## Default limits
The default limit is 1000 requests per minute per organisation and 100 requests per
second per API token. Bulk export endpoints have a separate limit of 10 requests per
minute.

## Exceeding the limit
Requests over the limit receive HTTP 429 with a Retry-After header giving the number
of seconds to wait. Clients should honour Retry-After rather than retrying
immediately. Sustained violation may result in temporary suspension of the token.

## Increasing limits
Rate limit increases are reviewed on request for accounts on the Business plan and
above. The review considers historical usage patterns and the stated reason. Typical
turnaround is two business days.
