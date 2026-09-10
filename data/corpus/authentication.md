# Authentication and Session Management

## Supported methods
The platform supports password authentication with TOTP two-factor, SAML 2.0 single
sign-on, and OIDC. Enterprise plans may enforce SSO for all members of an
organisation, which disables password login for those accounts.

## Session lifetime
Web sessions expire after 12 hours of inactivity or 7 days absolute, whichever comes
first. API tokens do not expire automatically but can be revoked. Revocation takes
effect within 60 seconds across all regions.

## Failed login handling
After five consecutive failed attempts the account is locked for 15 minutes. Lockout
events are written to the audit log. A password reset clears an active lockout.

## Error codes
Error E4021 indicates an expired session token and should be handled by
re-authenticating. Error E4033 indicates that SSO is enforced for the organisation
and password login was attempted. Error E4045 indicates the account is locked.
