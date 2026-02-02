# Common Attacks

**Know your enemy** | ⚠️ OWASP Top 10 | 🛡️ Prevention

---

## SQL Injection

=== "Attack"
    ```javascript
    // Vulnerable code
    const userId = req.query.id;
    const query = `SELECT * FROM users WHERE id = ${userId}`;
    // Attacker sends: id=1 OR 1=1
    // Executes: SELECT * FROM users WHERE id = 1 OR 1=1
    // Returns ALL users!
    ```

=== "Prevention"
    ```javascript
    // Use parameterized queries
    const query = 'SELECT * FROM users WHERE id = ?';
    db.query(query, [userId]);

    // Or ORM
    const user = await User.findById(userId);
    ```

---

## XSS (Cross-Site Scripting)

=== "Attack"
    ```html
    <!-- User input: <script>steal_cookies()</script> -->
    <!-- Rendered as: -->
    <div>Welcome <script>steal_cookies()</script></div>
    ```

=== "Prevention"
    ```javascript
    // Escape user input
    function escapeHTML(str) {
        return str.replace(/[&<>"']/g, (char) => {
            const escapeChars = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;'
            };
            return escapeChars[char];
        });
    }

    // Use Content Security Policy
    res.setHeader('Content-Security-Policy', "default-src 'self'");
    ```

---

## CSRF (Cross-Site Request Forgery)

=== "Attack"
    ```html
    <!-- Attacker's site -->
    <form action="https://bank.com/transfer" method="POST">
        <input name="to" value="attacker_account" />
        <input name="amount" value="10000" />
    </form>
    <script>document.forms[0].submit();</script>
    ```

=== "Prevention"
    ```javascript
    const csrf = require('csurf');
    app.use(csrf({ cookie: true }));

    // Include CSRF token in forms
    app.get('/form', (req, res) => {
        res.render('form', { csrfToken: req.csrfToken() });
    });
    ```

---

**Stay vigilant, stay secure! ⚠️**
