# CI/CD Pipelines

**Automate everything** | 🔄 Continuous Integration | 🚀 Continuous Deployment | ⚙️ Automation

---

## Overview

CI/CD automates the software delivery process from code commit to production deployment. It ensures code quality, reduces manual errors, and enables rapid, reliable releases.

**CI (Continuous Integration):** Automatically build and test code changes  
**CD (Continuous Delivery):** Automatically prepare releases for deployment  
**CD (Continuous Deployment):** Automatically deploy to production

---

## CI/CD Pipeline Stages

```
┌──────────┐   ┌───────┐   ┌──────┐   ┌──────┐   ┌────────┐   ┌─────────┐
│  Commit  │──▶│ Build │──▶│ Test │──▶│ Scan │──▶│ Package│──▶│ Deploy  │
│   Code   │   │       │   │      │   │      │   │        │   │         │
└──────────┘   └───────┘   └──────┘   └──────┘   └────────┘   └─────────┘
     ↓             ↓          ↓          ↓           ↓             ↓
   GitHub       Docker     Unit      Security    Docker        Kubernetes
              Compile     Tests       Scan       Registry      Production
```

---

## Complete Pipeline Example

=== "GitHub Actions"
    ```yaml
    name: CI/CD Pipeline

    on:
      push:
        branches: [main, develop]
      pull_request:
        branches: [main]

    env:
      DOCKER_REGISTRY: ghcr.io
      IMAGE_NAME: ${{ github.repository }}

    jobs:
      # Job 1: Build and Test
      build-and-test:
        runs-on: ubuntu-latest
        steps:
          - name: Checkout code
            uses: actions/checkout@v3

          - name: Set up Node.js
            uses: actions/setup-node@v3
            with:
              node-version: '18'
              cache: 'npm'

          - name: Install dependencies
            run: npm ci

          - name: Run linter
            run: npm run lint

          - name: Run unit tests
            run: npm test -- --coverage

          - name: Upload coverage
            uses: codecov/codecov-action@v3
            with:
              files: ./coverage/lcov.info

          - name: Build application
            run: npm run build

          - name: Run integration tests
            run: npm run test:integration

      # Job 2: Security Scan
      security-scan:
        runs-on: ubuntu-latest
        needs: build-and-test
        steps:
          - name: Checkout code
            uses: actions/checkout@v3

          - name: Run Snyk security scan
            uses: snyk/actions/node@master
            env:
              SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

          - name: Run SAST with SonarQube
            uses: sonarsource/sonarqube-scan-action@master
            env:
              SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
              SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}

      # Job 3: Build Docker Image
      build-image:
        runs-on: ubuntu-latest
        needs: [build-and-test, security-scan]
        if: github.ref == 'refs/heads/main'
        outputs:
          image-tag: ${{ steps.meta.outputs.tags }}
        steps:
          - name: Checkout code
            uses: actions/checkout@v3

          - name: Set up Docker Buildx
            uses: docker/setup-buildx-action@v2

          - name: Log in to GitHub Container Registry
            uses: docker/login-action@v2
            with:
              registry: ${{ env.DOCKER_REGISTRY }}
              username: ${{ github.actor }}
              password: ${{ secrets.GITHUB_TOKEN }}

          - name: Extract metadata
            id: meta
            uses: docker/metadata-action@v4
            with:
              images: ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}
              tags: |
                type=ref,event=branch
                type=sha,prefix={{branch}}-
                type=semver,pattern={{version}}

          - name: Build and push Docker image
            uses: docker/build-push-action@v4
            with:
              context: .
              push: true
              tags: ${{ steps.meta.outputs.tags }}
              labels: ${{ steps.meta.outputs.labels }}
              cache-from: type=registry,ref=${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:buildcache
              cache-to: type=registry,ref=${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:buildcache,mode=max

          - name: Scan Docker image
            uses: aquasecurity/trivy-action@master
            with:
              image-ref: ${{ steps.meta.outputs.tags }}
              format: 'sarif'
              output: 'trivy-results.sarif'

          - name: Upload Trivy results
            uses: github/codeql-action/upload-sarif@v2
            with:
              sarif_file: 'trivy-results.sarif'

      # Job 4: Deploy to Staging
      deploy-staging:
        runs-on: ubuntu-latest
        needs: build-image
        environment: staging
        steps:
          - name: Checkout code
            uses: actions/checkout@v3

          - name: Set up kubectl
            uses: azure/setup-kubectl@v3

          - name: Configure AWS credentials
            uses: aws-actions/configure-aws-credentials@v2
            with:
              aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
              aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
              aws-region: us-east-1

          - name: Update kubeconfig
            run: |
              aws eks update-kubeconfig --name staging-cluster --region us-east-1

          - name: Deploy to staging
            run: |
              kubectl set image deployment/myapp \
                myapp=${{ needs.build-image.outputs.image-tag }} \
                -n staging

          - name: Wait for rollout
            run: |
              kubectl rollout status deployment/myapp -n staging --timeout=5m

          - name: Run smoke tests
            run: |
              npm run test:smoke -- --env=staging

      # Job 5: Deploy to Production
      deploy-production:
        runs-on: ubuntu-latest
        needs: deploy-staging
        environment: production
        if: github.ref == 'refs/heads/main'
        steps:
          - name: Checkout code
            uses: actions/checkout@v3

          - name: Set up kubectl
            uses: azure/setup-kubectl@v3

          - name: Configure AWS credentials
            uses: aws-actions/configure-aws-credentials@v2
            with:
              aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
              aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
              aws-region: us-east-1

          - name: Update kubeconfig
            run: |
              aws eks update-kubeconfig --name production-cluster --region us-east-1

          - name: Deploy to production (canary)
            run: |
              # Deploy to canary subset
              kubectl apply -f k8s/canary-deployment.yaml
              
              # Wait and monitor
              sleep 300
              
              # Check metrics
              ERROR_RATE=$(curl -s http://prometheus/api/v1/query?query='error_rate')
              
              if [ "$ERROR_RATE" -lt "0.01" ]; then
                # Promote to full deployment
                kubectl apply -f k8s/production-deployment.yaml
              else
                # Rollback
                kubectl rollout undo deployment/myapp -n production
                exit 1
              fi

          - name: Notify Slack
            uses: 8398a7/action-slack@v3
            with:
              status: ${{ job.status }}
              text: 'Production deployment completed!'
              webhook_url: ${{ secrets.SLACK_WEBHOOK }}
    ```

=== "GitLab CI"
    ```yaml
    stages:
      - build
      - test
      - security
      - package
      - deploy

    variables:
      DOCKER_DRIVER: overlay2
      DOCKER_TLS_CERTDIR: "/certs"

    # Build stage
    build:
      stage: build
      image: node:18
      cache:
        paths:
          - node_modules/
      script:
        - npm ci
        - npm run build
      artifacts:
        paths:
          - dist/
        expire_in: 1 hour

    # Test stage
    unit-test:
      stage: test
      image: node:18
      dependencies:
        - build
      script:
        - npm run test:unit -- --coverage
      coverage: '/Statements\s*:\s*(\d+\.\d+)%/'
      artifacts:
        reports:
          coverage_report:
            coverage_format: cobertura
            path: coverage/cobertura-coverage.xml

    integration-test:
      stage: test
      image: node:18
      services:
        - postgres:14
        - redis:7
      variables:
        POSTGRES_DB: test_db
        POSTGRES_USER: test_user
        POSTGRES_PASSWORD: test_password
      dependencies:
        - build
      script:
        - npm run test:integration

    # Security stage
    security-scan:
      stage: security
      image: aquasec/trivy:latest
      script:
        - trivy fs --exit-code 1 --severity HIGH,CRITICAL .

    sast:
      stage: security
      image: returntocorp/semgrep
      script:
        - semgrep --config=auto --json --output=semgrep-results.json .
      artifacts:
        reports:
          sast: semgrep-results.json

    # Package stage
    docker-build:
      stage: package
      image: docker:latest
      services:
        - docker:dind
      before_script:
        - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
      script:
        - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
        - docker tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE:latest
        - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
        - docker push $CI_REGISTRY_IMAGE:latest
      only:
        - main

    # Deploy stage
    deploy-staging:
      stage: deploy
      image: bitnami/kubectl:latest
      environment:
        name: staging
        url: https://staging.example.com
      script:
        - kubectl set image deployment/myapp myapp=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -n staging
        - kubectl rollout status deployment/myapp -n staging
      only:
        - main

    deploy-production:
      stage: deploy
      image: bitnami/kubectl:latest
      environment:
        name: production
        url: https://example.com
      when: manual
      script:
        - kubectl set image deployment/myapp myapp=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -n production
        - kubectl rollout status deployment/myapp -n production
      only:
        - main
    ```

=== "Jenkins Pipeline"
    ```groovy
    pipeline {
        agent any
        
        environment {
            DOCKER_REGISTRY = 'docker.io'
            IMAGE_NAME = 'mycompany/myapp'
            DOCKER_CREDENTIALS = credentials('docker-hub-credentials')
            KUBECONFIG = credentials('kubeconfig-production')
        }
        
        stages {
            stage('Checkout') {
                steps {
                    checkout scm
                }
            }
            
            stage('Build') {
                steps {
                    script {
                        sh 'npm ci'
                        sh 'npm run build'
                    }
                }
            }
            
            stage('Test') {
                parallel {
                    stage('Unit Tests') {
                        steps {
                            sh 'npm run test:unit'
                        }
                    }
                    stage('Integration Tests') {
                        steps {
                            sh 'npm run test:integration'
                        }
                    }
                    stage('Lint') {
                        steps {
                            sh 'npm run lint'
                        }
                    }
                }
            }
            
            stage('Security Scan') {
                steps {
                    script {
                        sh 'npm audit --audit-level=high'
                        sh 'snyk test --severity-threshold=high'
                    }
                }
            }
            
            stage('Build Docker Image') {
                when {
                    branch 'main'
                }
                steps {
                    script {
                        def imageTag = "${env.BUILD_NUMBER}-${env.GIT_COMMIT.take(7)}"
                        docker.build("${IMAGE_NAME}:${imageTag}")
                        docker.build("${IMAGE_NAME}:latest")
                    }
                }
            }
            
            stage('Push Docker Image') {
                when {
                    branch 'main'
                }
                steps {
                    script {
                        docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-hub-credentials') {
                            def imageTag = "${env.BUILD_NUMBER}-${env.GIT_COMMIT.take(7)}"
                            docker.image("${IMAGE_NAME}:${imageTag}").push()
                            docker.image("${IMAGE_NAME}:latest").push()
                        }
                    }
                }
            }
            
            stage('Deploy to Staging') {
                when {
                    branch 'main'
                }
                steps {
                    script {
                        def imageTag = "${env.BUILD_NUMBER}-${env.GIT_COMMIT.take(7)}"
                        sh """
                            kubectl set image deployment/myapp \
                                myapp=${IMAGE_NAME}:${imageTag} \
                                -n staging
                            kubectl rollout status deployment/myapp -n staging
                        """
                    }
                }
            }
            
            stage('Smoke Tests') {
                when {
                    branch 'main'
                }
                steps {
                    sh 'npm run test:smoke -- --env=staging'
                }
            }
            
            stage('Deploy to Production') {
                when {
                    branch 'main'
                }
                input {
                    message "Deploy to production?"
                    ok "Deploy"
                }
                steps {
                    script {
                        def imageTag = "${env.BUILD_NUMBER}-${env.GIT_COMMIT.take(7)}"
                        sh """
                            kubectl set image deployment/myapp \
                                myapp=${IMAGE_NAME}:${imageTag} \
                                -n production
                            kubectl rollout status deployment/myapp -n production
                        """
                    }
                }
            }
        }
        
        post {
            always {
                junit 'test-results/**/*.xml'
                publishHTML([
                    reportDir: 'coverage',
                    reportFiles: 'index.html',
                    reportName: 'Coverage Report'
                ])
            }
            success {
                slackSend(
                    color: 'good',
                    message: "Build ${env.BUILD_NUMBER} succeeded!"
                )
            }
            failure {
                slackSend(
                    color: 'danger',
                    message: "Build ${env.BUILD_NUMBER} failed!"
                )
            }
        }
    }
    ```

---

## Testing Strategies

=== "Test Pyramid"
    ```
         /\
        /  \
       / E2E\      ← Few (slow, expensive)
      /______\
     /        \
    /Integration\   ← Some (medium speed)
   /____________\
  /              \
 /  Unit Tests   \  ← Many (fast, cheap)
/__________________\

    70% Unit Tests
    20% Integration Tests
    10% E2E Tests
    ```

    **Implementation:**
    ```javascript
    // Unit test (fast)
    describe('calculateTotal', () => {
        it('should sum item prices', () => {
            const items = [
                { price: 10 },
                { price: 20 }
            ];
            expect(calculateTotal(items)).toBe(30);
        });
    });

    // Integration test (medium)
    describe('Order API', () => {
        it('should create order', async () => {
            const response = await request(app)
                .post('/api/orders')
                .send({ userId: 1, items: [...] });
            
            expect(response.status).toBe(201);
            const order = await db.orders.findOne(response.body.id);
            expect(order).toBeDefined();
        });
    });

    // E2E test (slow)
    describe('Checkout flow', () => {
        it('should complete purchase', async () => {
            await page.goto('/products');
            await page.click('[data-testid="add-to-cart"]');
            await page.click('[data-testid="checkout"]');
            await page.fill('[name="card"]', '4242424242424242');
            await page.click('[data-testid="submit"]');
            
            await expect(page).toHaveURL('/order-confirmation');
        });
    });
    ```

=== "Test Coverage"
    ```yaml
    # Jest configuration
    {
      "coverageThreshold": {
        "global": {
          "branches": 80,
          "functions": 80,
          "lines": 80,
          "statements": 80
        },
        "critical": {
          "branches": 90,
          "functions": 90,
          "lines": 90,
          "statements": 90
        }
      }
    }
    ```

    **CI enforcement:**
    ```bash
    # Fail build if coverage drops
    npm test -- --coverage --coverageThreshold='{"global":{"lines":80}}'
    ```

---

## Security in CI/CD

=== "SAST (Static Analysis)"
    ```yaml
    # SonarQube scan
    sonarqube-scan:
      stage: security
      script:
        - sonar-scanner \
            -Dsonar.projectKey=myapp \
            -Dsonar.sources=src \
            -Dsonar.host.url=$SONAR_HOST_URL \
            -Dsonar.login=$SONAR_TOKEN
      only:
        - main
        - merge_requests
    ```

=== "DAST (Dynamic Analysis)"
    ```yaml
    # OWASP ZAP scan
    dast-scan:
      stage: security
      image: owasp/zap2docker-stable
      script:
        - zap-baseline.py \
            -t https://staging.example.com \
            -r zap-report.html
      artifacts:
        paths:
          - zap-report.html
    ```

=== "Dependency Scanning"
    ```yaml
    # Snyk dependency scan
    dependency-scan:
      stage: security
      script:
        - snyk test \
            --severity-threshold=high \
            --fail-on=all
        - snyk monitor
    ```

=== "Container Scanning"
    ```yaml
    # Trivy container scan
    container-scan:
      stage: security
      script:
        - trivy image \
            --severity HIGH,CRITICAL \
            --exit-code 1 \
            $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    ```

---

## Deployment Strategies in CI/CD

=== "Blue-Green"
    ```yaml
    deploy-production:
      script:
        # Deploy green
        - kubectl apply -f k8s/deployment-green.yaml
        - kubectl wait --for=condition=available deployment/myapp-green
        
        # Run smoke tests
        - ./scripts/smoke-test.sh green
        
        # Switch traffic
        - kubectl patch service myapp -p '{"spec":{"selector":{"version":"green"}}}'
        
        # Keep blue for rollback
        - echo "Blue deployment kept for 24h rollback window"
    ```

=== "Canary"
    ```yaml
    deploy-canary:
      script:
        # Deploy canary (10% traffic)
        - kubectl apply -f k8s/canary-10percent.yaml
        - sleep 300
        
        # Check metrics
        - ERROR_RATE=$(./scripts/check-metrics.sh)
        - |
          if [ "$ERROR_RATE" -gt "1" ]; then
            kubectl delete -f k8s/canary-10percent.yaml
            exit 1
          fi
        
        # Increase to 50%
        - kubectl apply -f k8s/canary-50percent.yaml
        - sleep 300
        
        # Full rollout
        - kubectl apply -f k8s/production-deployment.yaml
    ```

---

## Monitoring and Notifications

=== "Pipeline Metrics"
    ```
    Track:
    - Build duration
    - Test pass rate
    - Deployment frequency
    - Mean time to recovery (MTTR)
    - Change failure rate
    ```

=== "Notifications"
    ```yaml
    # Slack notification
    notify:
      script:
        - |
          curl -X POST $SLACK_WEBHOOK \
            -H 'Content-Type: application/json' \
            -d "{
              \"text\": \"Deployment to production completed!\",
              \"attachments\": [{
                \"color\": \"good\",
                \"fields\": [
                  {\"title\": \"Environment\", \"value\": \"production\"},
                  {\"title\": \"Version\", \"value\": \"$CI_COMMIT_SHA\"},
                  {\"title\": \"Deployed by\", \"value\": \"$GITLAB_USER_NAME\"}
                ]
              }]
            }"
    ```

---

## Best Practices

### ✅ **Do's**

1. **Keep pipelines fast**
   ```
   - Run tests in parallel
   - Cache dependencies
   - Use incremental builds
   Target: < 10 minutes from commit to deploy
   ```

2. **Fail fast**
   ```yaml
   # Run cheap tests first
   stages:
     - lint          # 30 seconds
     - unit-test     # 2 minutes
     - build         # 3 minutes
     - integration   # 5 minutes
     - deploy        # 10 minutes
   ```

3. **Version everything**
   ```bash
   # Tag images with:
   - Git commit SHA
   - Build number
   - Semantic version
   
   myapp:v1.2.3
   myapp:build-1234
   myapp:abc123f
   ```

### ❌ **Don'ts**

1. **Don't skip tests**
   - Every commit must pass tests
   - No "test later" mentality

2. **Don't store secrets in code**
   ```bash
   # Use secret management
   - GitHub Secrets
   - GitLab CI/CD variables
   - HashiCorp Vault
   - AWS Secrets Manager
   ```

3. **Don't deploy untested code**
   - Staging must pass before production
   - Smoke tests are mandatory

---

## Interview Talking Points

**Q: What's the difference between Continuous Delivery and Continuous Deployment?**

✅ **Strong Answer:**
> "Continuous Delivery means code is automatically tested and prepared for production deployment, but the actual deployment requires manual approval. Continuous Deployment goes further - every change that passes automated tests is automatically deployed to production without human intervention. I'd use Continuous Delivery for critical systems like banking where we want manual verification before production, and Continuous Deployment for less critical services like internal tools or feature-flagged releases where we can safely deploy multiple times per day."

**Q: How do you ensure pipeline security?**

✅ **Strong Answer:**
> "I'd implement multiple security layers: SAST tools like SonarQube scan code for vulnerabilities, dependency scanners like Snyk check for known CVEs in libraries, container scanners like Trivy inspect Docker images, and DAST tools like OWASP ZAP test running applications. I'd fail the build if high or critical vulnerabilities are found. For secrets, I'd use secret management tools and never store credentials in code. I'd also implement least-privilege access for CI/CD service accounts and audit all pipeline changes."

---

## Related Topics

- [Deployment Strategies](strategies.md) - Blue-green, canary deployments
- [Containers](containers.md) - Docker and containerization
- [Infrastructure as Code](infrastructure.md) - Terraform, CloudFormation
- [Monitoring](../observability/monitoring.md) - Track deployments

---

**Automate everything, deploy with confidence! 🚀**
