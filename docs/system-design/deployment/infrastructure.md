# Infrastructure as Code

Infrastructure as Code (IaC) is the practice of managing and provisioning computing resources through machine-readable configuration files rather than manual processes. Instead of logging into a cloud console and clicking through wizards, engineers define their entire infrastructure -- servers, networks, databases, load balancers -- as versioned text files that can be reviewed, tested, and applied automatically.

This shift from manual operations to code-driven infrastructure is one of the most consequential changes in modern operations. Netflix manages over 100,000 instances across AWS using IaC. Spotify provisions its entire backend infrastructure through Terraform, enabling hundreds of autonomous squads to deploy independently without stepping on each other.

## IaC Tools Comparison

| Tool | Approach | Language | Multi-Cloud | State Mgmt | Best For |
|------|----------|----------|-------------|------------|----------|
| **Terraform** | Declarative | HCL | Yes (any provider) | External state file | Multi-cloud, vendor-agnostic |
| **CloudFormation** | Declarative | YAML/JSON | AWS only | Managed by AWS | AWS-native shops |
| **AWS CDK** | Imperative | TypeScript, Python, Java | AWS only | Via CloudFormation | Developers who prefer real languages |
| **Pulumi** | Imperative | TypeScript, Python, Go, C# | Yes (any provider) | Managed or self-hosted | Multi-cloud with real languages |
| **Ansible** | Procedural | YAML | Yes | Stateless | Configuration management |

---

=== "Why IaC"

    ## The Case for Infrastructure as Code

    The traditional approach to infrastructure -- logging into servers, running manual commands, clicking through cloud consoles -- worked when teams managed a handful of machines. At scale, manual processes become the single greatest source of outages and security incidents.

    Consider a real scenario: a team at a mid-size fintech company manually configures their production environment across 40 EC2 instances, 3 RDS databases, and a dozen security groups. When a critical vulnerability requires patching every server's network configuration, an engineer spends 6 hours making changes by hand, accidentally misconfigures one security group, and exposes a database port to the public internet for 3 hours before anyone notices.

    With IaC, that same change is a one-line edit in a configuration file, reviewed by a teammate in a pull request, applied atomically across all resources in minutes, and trivially rolled back if something goes wrong.

    ### Manual vs IaC: A Side-by-Side Scenario

    ```
    SCENARIO: Provision identical staging environment

    Manual Process (Team of 3, ~2 days):
    +--------------------------------------------------+
    |  Day 1: Engineer A creates VPC, subnets           |
    |         Engineer B provisions RDS (wrong version!) |
    |         Engineer C sets up load balancer            |
    |  Day 2: Debug mismatched configs                   |
    |         Fix security group that was missed          |
    |         Document what was done (partially)          |
    |  Result: "Works" but differs from prod              |
    +--------------------------------------------------+

    IaC Process (1 engineer, ~30 minutes):
    +--------------------------------------------------+
    |  Step 1: terraform workspace new staging           |
    |  Step 2: terraform apply -var="env=staging"        |
    |  Result: Exact replica of production               |
    +--------------------------------------------------+
    ```

    ### Reproducibility

    When infrastructure is code, creating a new environment is no different from deploying an application. Airbnb uses IaC to spin up complete replica environments for each feature branch, letting engineers test against production-like infrastructure before merging. This eliminates the classic "works on my machine" problem extended to the infrastructure layer.

    ### Version Control and Audit Trail

    Every infrastructure change flows through Git -- who changed what, when, and why. This is not just good practice; regulations like SOC 2 and PCI DSS require auditable change histories. Capital One adopted Terraform specifically to satisfy regulatory requirements, replacing manual change tickets with pull request histories that auditors can review directly.

    ### Drift Detection

    Infrastructure drift occurs when the actual state of resources diverges from the defined state, typically because someone made a manual change. Terraform's `plan` command compares the desired state in code against real-world resources, surfacing any drift before it causes problems. HashiCorp reports that enterprises typically discover 15-30% of their resources have drifted from their defined state when they first adopt IaC.

    ### Disaster Recovery

    If an entire region goes down, IaC turns recovery from a multi-day scramble into a parameterized deployment. When AWS's us-east-1 experienced extended outages, companies with IaC-managed infrastructure redirected traffic and rebuilt in us-west-2 within hours. Companies relying on manual processes took days.

=== "Terraform"

    ## Terraform

    Terraform, created by HashiCorp, has become the de facto standard for multi-cloud infrastructure management. It uses a declarative model: you describe the desired end state, and Terraform figures out how to get there. Over 3,000 providers exist in the Terraform Registry, covering every major cloud and hundreds of SaaS services.

    ### Declarative Model

    The declarative approach means you never write step-by-step instructions. Instead, you declare "I want 3 servers behind a load balancer in us-east-1" and Terraform determines the order of operations, handles dependencies, and parallelizes where possible. If you later change the count to 5, Terraform adds exactly 2 servers without touching the existing 3.

    ```
    Declarative Workflow:

    +------------------+     +------------------+     +------------------+
    | Define desired   |---->| Terraform builds |---->| Terraform applies|
    | state in .tf     |     | dependency graph |     | changes in order |
    | files            |     | automatically    |     | (parallel where  |
    |                  |     |                  |     |  safe)           |
    +------------------+     +------------------+     +------------------+
    ```

    ### State Management

    Terraform maintains a state file that maps your configuration to real-world resources. This file is the source of truth for what Terraform manages. When you run `terraform plan`, Terraform compares your config against the state file and the actual cloud resources, producing a precise diff of what will change.

    State is critically important and notoriously tricky. The state file contains sensitive data (resource IDs, IP addresses, sometimes passwords), so it must be stored securely. In production, teams store state remotely in S3, GCS, or Terraform Cloud rather than locally.

    ### Plan/Apply Workflow

    The plan/apply workflow is Terraform's safety net. The `plan` phase is a dry run that shows exactly what will be created, modified, or destroyed without making any changes. The `apply` phase executes the plan. This two-step process catches mistakes before they reach production -- a misconfigured security group shows up in the plan diff, not as a production incident.

    ```hcl
    # Minimal Terraform: define a resource and its dependencies
    resource "aws_instance" "web" {
      ami           = "ami-0c55b159cbfafe1f0"
      instance_type = "t3.micro"
      subnet_id     = aws_subnet.public.id
      tags = { Name = "web-server" }
    }
    ```

    ### Multi-Cloud Advantage

    Terraform's greatest differentiator is provider-agnostic infrastructure. A single Terraform codebase can provision AWS networking, deploy containers to GCP's GKE, configure Cloudflare DNS, and set up Datadog monitoring. Shopify uses Terraform to manage infrastructure spanning AWS and GCP, keeping a consistent workflow regardless of which cloud hosts a particular service. This avoids deep vendor lock-in and allows teams to choose the best service from any provider.

    ### Modules

    Terraform modules are reusable packages of infrastructure configuration. Rather than copying VPC definitions across 20 projects, teams create a `vpc` module once and reference it everywhere. The Terraform Registry hosts thousands of community modules -- for example, the official AWS VPC module has been downloaded over 50 million times. Internally, companies like Uber maintain private module registries, enforcing security and compliance standards through shared infrastructure building blocks.

=== "Cloud-Native"

    ## Cloud-Native IaC Tools

    While Terraform dominates the multi-cloud space, cloud providers offer their own IaC tools with deeper integration into their ecosystems. The choice between vendor-native and vendor-agnostic tooling is one of the most common infrastructure decisions teams face.

    ### AWS CloudFormation

    CloudFormation is AWS's native IaC service. It uses YAML or JSON templates to define AWS resources, and AWS manages the entire lifecycle -- state tracking, rollback on failure, and drift detection are built in. Because CloudFormation is an AWS service, it supports new AWS features on launch day, while Terraform providers sometimes lag by weeks or months.

    The main drawback is verbosity. A CloudFormation template for a simple VPC with subnets can easily reach 200+ lines of YAML. CloudFormation also lacks the rich module ecosystem that Terraform enjoys, and it is locked to AWS exclusively.

    ### AWS CDK (Cloud Development Kit)

    The CDK addresses CloudFormation's verbosity problem by letting developers define infrastructure using familiar programming languages. Under the hood, CDK synthesizes CloudFormation templates, so you get all of CloudFormation's reliability with the expressiveness of TypeScript, Python, or Java.

    The type-safety advantage is significant. In CDK with TypeScript, your IDE catches invalid property names, wrong resource types, and incompatible configurations at compile time -- errors that in CloudFormation or Terraform would only surface at deploy time. AWS reports that CDK users ship infrastructure changes 30-40% faster than raw CloudFormation users because of this tighter feedback loop.

    ```typescript
    // CDK: type-safe infrastructure in 5 lines
    const vpc = new ec2.Vpc(this, 'AppVpc', { maxAzs: 2 });
    const cluster = new ecs.Cluster(this, 'Cluster', { vpc });
    const service = new ecsPatterns.ApplicationLoadBalancedFargateService(
      this, 'Service', { cluster, taskImageOptions: { image: ecs.ContainerImage.fromRegistry('nginx') } }
    );
    ```

    ### Pulumi

    Pulumi takes the CDK's "real languages" approach and extends it across all major clouds. Like CDK, you write infrastructure in TypeScript, Python, Go, or C#. Unlike CDK, Pulumi works with AWS, GCP, Azure, and Kubernetes natively. Pulumi manages its own state (hosted or self-managed) rather than generating CloudFormation templates.

    Companies that need multi-cloud support but dislike HCL syntax gravitate toward Pulumi. Snowflake uses Pulumi to manage infrastructure across AWS and Azure with consistent TypeScript code.

    ### When to Use Which

    | Scenario | Recommended Tool | Why |
    |----------|-----------------|-----|
    | Multi-cloud, large team | Terraform | Largest ecosystem, most hiring pool |
    | AWS-only, ops-focused team | CloudFormation | Zero state management, native support |
    | AWS-only, developer-heavy team | AWS CDK | Type safety, familiar languages |
    | Multi-cloud, TypeScript/Go team | Pulumi | Real languages + multi-cloud |
    | Configuration management | Ansible | Agentless, procedural tasks |

=== "Best Practices"

    ## IaC Best Practices

    The tools themselves are only half the story. How teams organize, secure, and operate their IaC determines whether it becomes a reliable foundation or a tangled mess.

    ### Remote State with Locking

    Terraform state must be stored remotely for any team larger than one person. When two engineers run `terraform apply` simultaneously against local state files, the result is corrupted state and orphaned resources. Remote backends like S3 (with DynamoDB locking) or Terraform Cloud ensure only one operation runs at a time.

    ```
    Remote State Architecture:

    Engineer A                          Engineer B
        |                                   |
        v                                   v
    terraform apply                   terraform apply
        |                                   |
        +---------->  S3 Bucket  <----------+
                     (state file)
                         |
                    DynamoDB Table
                    (lock record)
                         |
              "Lock acquired by A,
               B must wait..."
    ```

    HashiCorp's recommended workflow is: store state in a shared backend, enable locking, encrypt state at rest, and restrict access via IAM policies. Never commit state files to Git -- they contain sensitive data and will cause merge conflicts.

    ### Module Reuse and Composition

    Well-structured IaC projects follow the same principles as well-structured application code: DRY (Don't Repeat Yourself), clear interfaces, and separation of concerns. Teams should build a library of internal modules for common patterns (VPCs, ECS clusters, RDS databases) with standardized inputs and outputs.

    A proven structure used by companies like Gruntwork and recommended by HashiCorp separates modules from their instantiation:

    ```
    Infrastructure Repository Layout:

    infrastructure/
    +-- modules/              <-- Reusable building blocks
    |   +-- networking/       (VPC, subnets, NAT gateways)
    |   +-- compute/          (EC2, ASG, launch templates)
    |   +-- database/         (RDS, ElastiCache)
    |   +-- monitoring/       (CloudWatch, alarms)
    +-- environments/         <-- Per-environment configs
    |   +-- dev/
    |   |   +-- main.tf       (references modules)
    |   |   +-- variables.tf  (dev-specific values)
    |   +-- staging/
    |   +-- production/
    +-- .github/workflows/    <-- CI/CD for infrastructure
    ```

    ### Environment Management

    The cardinal rule of environment management is that dev, staging, and production should use identical infrastructure definitions with different parameters. The only differences should be scale (smaller instances in dev), cost (fewer replicas in staging), and access controls. When environments diverge structurally, staging stops catching the bugs it exists to catch.

    Terraform workspaces or separate state files per environment both work. The key is that promoting from staging to production should require changing zero infrastructure code -- only variable values change.

    ### GitOps for Infrastructure

    GitOps extends the pull request workflow to infrastructure changes. No one runs `terraform apply` from their laptop. Instead, all changes go through pull requests, CI runs `terraform plan` and posts the output as a PR comment, and merging to main triggers the actual apply via CI/CD.

    ```
    GitOps Workflow:

    Developer          GitHub             CI/CD              Cloud
       |                  |                  |                  |
       |-- push branch -->|                  |                  |
       |-- open PR ------>|                  |                  |
       |                  |-- trigger plan ->|                  |
       |                  |<- post plan -----|                  |
       |                  |   as PR comment  |                  |
       |-- review plan -->|                  |                  |
       |-- merge PR ----->|                  |                  |
       |                  |-- trigger apply->|                  |
       |                  |                  |-- provision ---->|
       |                  |                  |<- confirm -------|
       |                  |<- update status -|                  |
    ```

    Atlassian, GitHub, and Stripe all enforce this pattern: no manual applies, every infrastructure change has a reviewable plan, and the Git log serves as the complete audit trail. This practice alone eliminates the majority of infrastructure-related outages caused by ad-hoc changes.

---

## Key Takeaways

Infrastructure as Code transforms infrastructure from a fragile, manually maintained artifact into a versioned, testable, and reproducible system. The choice of tool matters less than the discipline of treating infrastructure with the same rigor as application code: version control, code review, automated testing, and CI/CD pipelines.

For teams starting out, Terraform offers the broadest applicability and largest community. AWS-only teams should seriously consider CDK for its type safety and developer experience. Regardless of tool choice, remote state management, module reuse, and GitOps workflows are non-negotiable practices for production infrastructure.

## Related Topics

- [Containers](containers.md) - Container orchestration and runtime infrastructure
- [CI/CD](ci-cd.md) - Automating infrastructure deployment pipelines
- [Deployment Strategies](strategies.md) - Blue-green, canary, and rolling deployments
