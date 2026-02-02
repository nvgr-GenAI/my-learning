# Authorization

**Control who can do what** | 🛡️ RBAC | 🎯 ABAC | 🔒 Permissions

---

## Overview

Authorization determines what an authenticated user is allowed to do. It's about access control, not identity verification.

**Authentication vs Authorization:**
- **Authentication:** Who are you? (Identity)
- **Authorization:** What can you do? (Permissions)

---

## Authorization Models

| Model | Complexity | Flexibility | Use Case |
|-------|-----------|-------------|----------|
| **ACL** (Access Control List) | Low | Low | Simple file permissions |
| **RBAC** (Role-Based) | Medium | Medium | Most applications |
| **ABAC** (Attribute-Based) | High | High | Complex enterprise systems |

---

## RBAC (Role-Based Access Control)

=== "Concept"
    **Assign permissions to roles, not users**

    ```
    User → has → Role → has → Permissions

    Example:
    Alice → Admin → can create/read/update/delete users
    Bob → Editor → can create/read/update posts
    Charlie → Viewer → can read posts
    ```

=== "Implementation"
    ```javascript
    // Database schema
    const roles = {
        admin: {
            permissions: ['users:create', 'users:read', 'users:update', 'users:delete',
                         'posts:create', 'posts:read', 'posts:update', 'posts:delete']
        },
        editor: {
            permissions: ['posts:create', 'posts:read', 'posts:update',
                         'users:read']
        },
        viewer: {
            permissions: ['posts:read']
        }
    };

    // Authorization middleware
    function authorize(permission) {
        return (req, res, next) => {
            const user = req.user;
            
            if (!user) {
                return res.status(401).json({ error: 'Not authenticated' });
            }

            const role = roles[user.role];
            
            if (!role || !role.permissions.includes(permission)) {
                return res.status(403).json({ error: 'Forbidden' });
            }

            next();
        };
    }

    // Usage
    app.post('/users', authorize('users:create'), async (req, res) => {
        // Create user
    });

    app.get('/posts', authorize('posts:read'), async (req, res) => {
        // List posts
    });

    app.delete('/users/:id', authorize('users:delete'), async (req, res) => {
        // Delete user
    });
    ```

=== "Hierarchical Roles"
    ```javascript
    // Role hierarchy
    const roleHierarchy = {
        superadmin: ['admin', 'editor', 'viewer'],
        admin: ['editor', 'viewer'],
        editor: ['viewer'],
        viewer: []
    };

    function hasPermission(user, permission) {
        const userRole = user.role;
        const applicableRoles = [userRole, ...roleHierarchy[userRole]];
        
        for (const role of applicableRoles) {
            if (roles[role]?.permissions.includes(permission)) {
                return true;
            }
        }
        
        return false;
    }

    // Usage
    app.delete('/posts/:id', (req, res) => {
        if (!hasPermission(req.user, 'posts:delete')) {
            return res.status(403).json({ error: 'Forbidden' });
        }
        
        // Delete post
    });
    ```

---

## ABAC (Attribute-Based Access Control)

=== "Concept"
    **Make decisions based on attributes**

    ```
    Attributes:
    - User attributes: role, department, location
    - Resource attributes: owner, classification, createdAt
    - Environment: time, IP address, device
    - Action: read, write, delete

    Policy:
    "Users can edit documents if they are the owner OR
     if they are in the same department AND document is not classified"
    ```

=== "Implementation"
    ```javascript
    // Policy engine
    class PolicyEngine {
        constructor() {
            this.policies = [];
        }

        addPolicy(policy) {
            this.policies.push(policy);
        }

        evaluate(user, resource, action, environment = {}) {
            for (const policy of this.policies) {
                const result = policy.evaluate(user, resource, action, environment);
                if (result === 'allow') return true;
                if (result === 'deny') return false;
            }
            return false; // Default deny
        }
    }

    // Example policies
    const canEditDocument = {
        evaluate: (user, resource, action) => {
            if (action !== 'edit') return 'continue';

            // Owner can always edit
            if (resource.ownerId === user.id) {
                return 'allow';
            }

            // Same department can edit non-classified docs
            if (user.department === resource.department &&
                resource.classification !== 'confidential') {
                return 'allow';
            }

            return 'deny';
        }
    };

    const timeBasedAccess = {
        evaluate: (user, resource, action, environment) => {
            const hour = new Date().getHours();
            
            // Restrict access outside business hours for non-admins
            if (user.role !== 'admin' && (hour < 9 || hour > 17)) {
                return 'deny';
            }

            return 'continue';
        }
    };

    // Usage
    const policyEngine = new PolicyEngine();
    policyEngine.addPolicy(timeBasedAccess);
    policyEngine.addPolicy(canEditDocument);

    app.put('/documents/:id', async (req, res) => {
        const document = await db.documents.findById(req.params.id);
        
        const allowed = policyEngine.evaluate(
            req.user,
            document,
            'edit',
            { ip: req.ip, time: new Date() }
        );

        if (!allowed) {
            return res.status(403).json({ error: 'Forbidden' });
        }

        // Update document
    });
    ```

---

## Resource-Based Authorization

=== "Ownership"
    ```javascript
    // Check if user owns the resource
    app.delete('/posts/:id', async (req, res) => {
        const post = await db.posts.findById(req.params.id);
        
        if (!post) {
            return res.status(404).json({ error: 'Post not found' });
        }

        // Only owner or admin can delete
        if (post.authorId !== req.user.id && req.user.role !== 'admin') {
            return res.status(403).json({ error: 'Forbidden' });
        }

        await db.posts.delete(req.params.id);
        res.status(204).send();
    });
    ```

=== "Scope-Based"
    ```javascript
    // Limit access to user's own data
    app.get('/orders', async (req, res) => {
        let query = {};

        // Non-admins can only see their own orders
        if (req.user.role !== 'admin') {
            query.userId = req.user.id;
        }

        const orders = await db.orders.find(query);
        res.json(orders);
    });
    ```

---

## Interview Talking Points

**Q: RBAC vs ABAC - when to use each?**

✅ **Strong Answer:**
> "I'd use RBAC for most applications because it's simpler and covers 90% of use cases - like an e-commerce site with admins, sellers, and customers. Each role has clear permissions. However, I'd use ABAC for complex enterprise scenarios where permissions depend on multiple factors. For example, a healthcare system where doctors can only access patient records from their department, during their shift, for patients they're treating. ABAC's attribute-based policies handle this naturally, while RBAC would require creating hundreds of specific roles."

---

**Least privilege: grant minimum permissions needed! 🛡️**
