# Common Attacks

Understanding how attacks work is essential for building defenses. This isn't about learning to attack — it's about recognizing the patterns that attackers exploit so you can design systems that don't have those vulnerabilities in the first place. The attacks below represent the most common and impactful threats to web applications, drawn from the OWASP Top 10.

---

=== "SQL Injection"

    ## SQL Injection

    The most dangerous and most preventable attack. SQL injection occurs when user input is concatenated directly into a SQL query, allowing an attacker to modify the query's logic.

    ```
    Vulnerable:
      query = "SELECT * FROM users WHERE id = " + user_input

      Normal input: 42
      → SELECT * FROM users WHERE id = 42  (works fine)

      Malicious input: 42 OR 1=1
      → SELECT * FROM users WHERE id = 42 OR 1=1  (returns ALL users)

      Destructive input: 42; DROP TABLE users; --
      → SELECT * FROM users WHERE id = 42; DROP TABLE users; --  (deletes table)
    ```

    ### Prevention

    **Parameterized queries** (also called prepared statements) are the complete solution. The database treats parameters as data, never as SQL code — even if the input contains SQL syntax.

    ```
    Safe: query("SELECT * FROM users WHERE id = ?", [user_input])
    ```

    The `?` is a placeholder. The database engine processes the query structure first, then inserts the parameter value as pure data. There is no way for the input to alter the query structure.

    **ORMs** (Sequelize, SQLAlchemy, ActiveRecord) use parameterized queries by default. As long as you use the ORM's query builder and avoid raw SQL with concatenated input, you're protected.

    In 2017, **Equifax** suffered a breach affecting 147 million people. While the initial vector was an unpatched vulnerability (Apache Struts), the attackers moved laterally through databases that had insufficient query parameterization and access controls.

=== "XSS"

    ## XSS (Cross-Site Scripting)

    XSS attacks inject malicious JavaScript into a web page that other users view. When the victim's browser renders the page, it executes the attacker's script with the victim's session and permissions.

    ### Types of XSS

    ```
    Stored XSS (persistent):
      Attacker posts a comment: "<script>steal_cookies()</script>"
      Comment stored in database
      Every user who views the page executes the script

    Reflected XSS:
      Attacker crafts a URL: example.com/search?q=<script>steal_cookies()</script>
      Server reflects the input in the page without escaping
      Victim clicks the link, script executes in their browser

    DOM-based XSS:
      Client-side JavaScript reads from URL/input and inserts into DOM
      document.getElementById('output').innerHTML = location.hash.slice(1)
      No server involvement — entirely client-side vulnerability
    ```

    **Stored XSS** is the most dangerous because it affects every user who views the page. In 2005, the **Samy worm** exploited stored XSS on MySpace — a self-replicating script that added the attacker as a friend and spread to over 1 million profiles in 20 hours.

    ### Prevention

    **Output encoding.** When inserting user-generated content into HTML, encode special characters: `<` becomes `&lt;`, `>` becomes `&gt;`, `"` becomes `&quot;`. Modern frameworks (React, Angular, Vue) do this automatically for content rendered through their template systems.

    **Content Security Policy (CSP).** An HTTP header that tells the browser which sources of JavaScript are allowed to execute. A strict CSP blocks inline scripts entirely:

    ```
    Content-Security-Policy: default-src 'self'; script-src 'self' https://trusted-cdn.com
    ```

    This means only scripts loaded from your domain or the trusted CDN can execute — even if an attacker injects a `<script>` tag, the browser won't run it.

    **httpOnly cookies.** Mark session cookies as httpOnly so JavaScript can't access them. Even if an XSS attack succeeds, it can't steal the session cookie.

=== "CSRF"

    ## CSRF (Cross-Site Request Forgery)

    CSRF tricks a user's browser into making a request to a site where they're already authenticated. Because browsers automatically include cookies with requests, the target site sees a valid authenticated request — it can't distinguish between the user's intentional action and the forged request.

    ```
    Attack scenario:

    1. User logs into bank.com (session cookie stored)

    2. User visits evil.com (in another tab)

    3. evil.com contains hidden form:
       <form action="https://bank.com/transfer" method="POST">
         <input name="to" value="attacker" />
         <input name="amount" value="10000" />
       </form>
       <script>document.forms[0].submit();</script>

    4. Browser sends POST to bank.com WITH the user's session cookie
       → Bank processes transfer because the session is valid
    ```

    ### Prevention

    **CSRF tokens.** The server generates a random token for each session and embeds it in forms as a hidden field. On submission, the server verifies the token matches. An attacker on a different site can't know the token value — the same-origin policy prevents them from reading it.

    **SameSite cookies.** Set `SameSite=Strict` or `SameSite=Lax` on session cookies. The browser won't send these cookies on cross-site requests, preventing CSRF entirely. This is the modern, preferred defense — Chrome, Firefox, and Safari all support it.

    **Check the Origin header.** On state-changing requests, verify that the `Origin` or `Referer` header matches your domain. Requests from `evil.com` will have `Origin: https://evil.com`, not your domain.

=== "Other Attacks"

    ## Other Critical Attacks

    ### Broken Authentication

    Not a single attack but a category: weak passwords allowed, no brute-force protection, session IDs in URLs, credentials transmitted over HTTP, predictable password reset tokens. The defense is systematic: enforce strong passwords, implement MFA, use secure session management, rate-limit login attempts.

    ### Broken Access Control

    When authorization checks are missing or flawed. A user changes `/api/users/123/orders` to `/api/users/456/orders` and sees another user's orders. The defense: check authorization on every request, at every level. See [Authorization](authorization.md).

    ### Security Misconfiguration

    Default credentials left in place, unnecessary services running, detailed error messages exposed to users, outdated software with known vulnerabilities. **The 2017 Equifax breach** started with an unpatched Apache Struts vulnerability that had a fix available for two months before the attack.

    ### Man-in-the-Middle (MITM)

    An attacker intercepts communication between two parties, reading or modifying traffic in transit. On unencrypted HTTP, this is trivial on public WiFi networks. The defense: **TLS everywhere**. See [Encryption](encryption.md).

---

## Defense Summary

| Attack | Primary Defense | Secondary Defense |
|---|---|---|
| **SQL Injection** | Parameterized queries | ORMs, input validation |
| **Stored XSS** | Output encoding | CSP headers, httpOnly cookies |
| **Reflected XSS** | Output encoding | CSP headers, input validation |
| **CSRF** | SameSite cookies | CSRF tokens, Origin header checks |
| **Broken Auth** | MFA, rate limiting | Strong password policy, secure sessions |
| **Broken Access Control** | Server-side authorization checks | Resource ownership verification |
| **MITM** | TLS/HTTPS | HSTS, certificate pinning |
| **Misconfiguration** | Automated scanning, updates | Minimal attack surface, no defaults |

---

## Key Takeaways

1. **Parameterized queries eliminate SQL injection completely.** There is no reason to ever concatenate user input into SQL. Use parameterized queries or ORMs — always.

2. **Modern frameworks prevent most XSS by default.** React, Angular, and Vue auto-encode output. The risk comes from bypassing these protections (dangerouslySetInnerHTML, v-html) or rendering raw HTML.

3. **SameSite cookies are the simplest CSRF defense.** Set `SameSite=Lax` on all session cookies. It's a one-line change that prevents the most common CSRF vectors.

4. **Security is about layers.** No single defense is perfect. Combine input validation, output encoding, authentication, authorization, encryption, and monitoring. An attacker who bypasses one layer should face another.

5. **Keep software updated.** Many of the largest breaches (Equifax, WannaCry) exploited known vulnerabilities with available patches. Automated dependency scanning and regular updates prevent a significant percentage of attacks.

---

## Related Topics

- **[Authentication](authentication.md)** — securing the login process
- **[Authorization](authorization.md)** — preventing unauthorized access
- **[Encryption](encryption.md)** — protecting data in transit and at rest
- **[API Security](api-security.md)** — rate limiting and input validation
