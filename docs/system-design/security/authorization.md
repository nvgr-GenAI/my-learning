# Authorization

Authentication answers "who are you?" Authorization answers the follow-up: **what are you allowed to do?** These are separate concerns, though they're often confused. A user can be authenticated (verified identity) but not authorized (not permitted) to perform a specific action.

Authorization is how you enforce rules like "only admins can delete users," "users can only see their own orders," and "contractors can access the project repository but not the production database." The challenge is designing a system flexible enough to express these rules without becoming so complex that nobody understands who can do what.

---

## Authorization Models

There are three primary models, each suited to different levels of complexity.

```
ACL (Access Control List):
  File X → [Alice: read/write, Bob: read, Everyone: none]

RBAC (Role-Based Access Control):
  Alice → Admin role → [create, read, update, delete] on all resources
  Bob → Viewer role → [read] on all resources

ABAC (Attribute-Based Access Control):
  IF user.department == resource.department
  AND user.clearance >= resource.classification
  AND time.hour BETWEEN 9 AND 17
  THEN allow
```

| Model | Complexity | Flexibility | Best For |
|---|---|---|---|
| **ACL** | Low | Low | File systems, simple resources |
| **RBAC** | Medium | Medium | Most web applications |
| **ABAC** | High | High | Enterprise, healthcare, government |

---

=== "RBAC"

    ## RBAC (Role-Based Access Control)

    RBAC is the workhorse of web application authorization. Instead of assigning permissions directly to users, you assign users to **roles**, and roles have **permissions**. This indirection simplifies management — when a new employee joins as an editor, you assign them the "editor" role, and they automatically get all editor permissions.

    ```
    User ──→ Role ──→ Permissions

    Alice ──→ Admin ──→ users:create, users:read, users:update, users:delete
                         posts:create, posts:read, posts:update, posts:delete

    Bob ──→ Editor ──→ posts:create, posts:read, posts:update
                        users:read

    Charlie ──→ Viewer ──→ posts:read
    ```

    ### Hierarchical Roles

    In many systems, roles form a hierarchy — an admin can do everything an editor can, plus more. This avoids duplicating permissions across roles:

    ```
    Super Admin
        └──→ Admin (inherits all below)
                └──→ Editor (inherits all below)
                        └──→ Viewer (base permissions)
    ```

    **AWS IAM** is one of the most sophisticated RBAC implementations in production. It uses policies attached to roles, with inheritance, conditions, and resource-level permissions. Google Cloud IAM follows a similar model with predefined roles (Viewer, Editor, Owner) plus custom roles.

    ### When RBAC Works Well

    - **Clear role boundaries.** E-commerce: admin, seller, customer. CMS: admin, editor, author, subscriber. SaaS: owner, admin, member, guest.
    - **Moderate number of roles.** RBAC works well with 5-15 roles. Beyond 50+ roles, it becomes hard to audit who has access to what.
    - **Permissions don't depend on context.** An admin can delete any user — it doesn't matter which user, what time it is, or what department they're in.

    ### When RBAC Breaks Down

    When you need rules like "doctors can only access records of patients in their department, during their shift, from approved devices" — that's three conditions beyond role membership. You'd need to create roles like "cardiology-doctor-day-shift-approved-device," which leads to **role explosion**. This is where ABAC takes over.

=== "ABAC"

    ## ABAC (Attribute-Based Access Control)

    ABAC makes authorization decisions based on **attributes** of the user, the resource, the action, and the environment. Instead of static role-permission mappings, you write policies that evaluate attributes dynamically.

    ### The Four Attribute Categories

    ```
    ┌─────────────────┐    ┌──────────────────┐
    │ User Attributes  │    │ Resource Attrs    │
    │ - role           │    │ - owner           │
    │ - department     │    │ - classification  │
    │ - clearance      │    │ - department      │
    │ - location       │    │ - created_date    │
    └────────┬────────┘    └────────┬─────────┘
             │                       │
             ▼                       ▼
        ┌──────────────────────────────┐
        │        Policy Engine         │
        │  "ALLOW if user.dept ==      │
        │   resource.dept AND          │
        │   user.clearance >= 3 AND    │
        │   env.time in business_hours"│
        └──────────────┬───────────────┘
             ▲                       ▲
             │                       │
    ┌────────┴────────┐    ┌────────┴─────────┐
    │ Action           │    │ Environment       │
    │ - read           │    │ - time of day     │
    │ - write          │    │ - IP address      │
    │ - delete         │    │ - device type     │
    └─────────────────┘    └──────────────────┘
    ```

    ### Real-World ABAC Examples

    **Healthcare (HIPAA compliance).** A hospital system where access depends on: doctor's specialty matching patient's department, active treatment relationship, access during scheduled shift hours, from a hospital-network device.

    **Financial services.** Trading systems where authorization depends on: trader's desk, trade amount vs. their limit, market hours, and whether compliance has flagged the counterparty.

    **Google's internal authorization** (Zanzibar) uses a relationship-based model that combines RBAC and ABAC concepts. It processes millions of authorization checks per second across Google's services — checking things like "does this user have edit access to this Google Doc?" which depends on direct sharing, folder permissions, domain-wide settings, and link sharing rules.

    ### The Trade-Off

    ABAC is powerful but complex. Every policy must be tested, audited, and understood by security teams. Debugging "why was this user denied access?" requires tracing through multiple attribute evaluations. Use ABAC when the authorization requirements genuinely demand it — not as a default choice.

=== "Resource-Based"

    ## Resource-Based Authorization

    A pattern that applies regardless of whether you use RBAC or ABAC: **ownership checks**. Users should only access their own resources unless they have explicit permission to access others.

    ```
    User requests DELETE /posts/42
        │
        ├─ Is user the owner of post 42? → Allow
        │
        ├─ Is user an admin? → Allow
        │
        └─ Neither? → 403 Forbidden
    ```

    This pattern appears in virtually every application:

    - **GitHub**: You can edit your own repos, and repos where you've been granted collaborator access. Organization admins can edit any org repo.
    - **Shopify**: Store owners see their own store's data. Shopify support staff have read-only access with audit logging.
    - **Stripe**: API keys are scoped to an account. One customer's API key cannot access another customer's data, regardless of role.

    ### Scope-Based Access

    Rather than all-or-nothing access, scope resources to the user's context:

    - Non-admin users requesting `GET /orders` only see their own orders — the query is automatically filtered by `user_id`
    - OAuth tokens carry scopes like `read:user` or `write:repos` that limit what the token holder can do, even if the underlying user has broader permissions
    - API keys can be scoped to specific resources or actions: read-only keys, write keys for specific collections

---

## Choosing an Authorization Model

```
Do permissions depend on more than the user's role?
    │
    ├─ No → RBAC
    │       (most web apps, SaaS platforms, CMSes)
    │
    └─ Yes → Do they depend on resource attributes?
              │
              ├─ Just ownership → RBAC + ownership checks
              │   (social media, e-commerce, multi-tenant SaaS)
              │
              └─ Multiple conditions → ABAC
                  (healthcare, finance, government, compliance-heavy systems)
```

---

## Key Takeaways

1. **RBAC covers 90% of applications.** Roles with permissions are simple to implement, easy to audit, and well-supported by every framework and cloud provider.

2. **Always check resource ownership.** Even with RBAC, verify that users can only access resources they own or have been explicitly granted access to. This prevents horizontal privilege escalation (user A accessing user B's data).

3. **ABAC is for genuine complexity.** If your authorization rules depend on time, location, resource classification, or relationships beyond simple ownership, ABAC is the right tool. Don't use it for simple role checks.

4. **Principle of least privilege.** Grant the minimum permissions needed. Start with no access and add permissions explicitly, rather than starting with full access and restricting.

5. **Authorization must be enforced server-side.** Client-side checks (hiding UI buttons) are for UX, not security. Every API endpoint must independently verify authorization.

---

## Related Topics

- **[Authentication](authentication.md)** — verifying identity before checking permissions
- **[API Security](api-security.md)** — rate limiting and API key management
- **[Common Attacks](common-attacks.md)** — privilege escalation and injection attacks
