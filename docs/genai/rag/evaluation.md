# RAG Evaluation: Measuring Success

!!! tip "📊 What Gets Measured Gets Improved"
    Building RAG is one thing, but knowing if it's working well is another! Let's explore how to evaluate and improve your RAG systems systematically.

## 🎯 The Evaluation Challenge

### 📖 Why RAG Evaluation is Tricky

=== "🤔 The Complexity Problem"

    **Traditional ML Evaluation:**
    ```
    Input → Model → Output → Compare with Ground Truth → Score
    ```
    
    **RAG Evaluation:**
    ```
    Question → Retrieval → Context → Generation → Answer
         ↓         ↓          ↓         ↓         ↓
      Relevance? Quality?  Accuracy? Fluency? Faithfulness?
    ```
    
    **Multiple Dimensions to Evaluate:**
    - 🔍 **Retrieval Quality**: Did we find relevant documents?
    - 📝 **Generation Quality**: Is the answer well-written?
    - ✅ **Factual Accuracy**: Is the information correct?
    - 🎯 **Relevance**: Does it answer the question?
    - 📚 **Faithfulness**: Is it based on retrieved context?

=== "🎭 Real-World Challenges"

    **The Human Problem:**
    - 👥 **Subjective judgments**: What's "good" varies by person
    - 🌍 **Domain expertise**: Technical accuracy needs experts
    - 💰 **Expensive annotation**: Human evaluation is costly
    - ⏰ **Time-consuming**: Manual review doesn't scale
    
    **The Context Problem:**
    - 📚 **Partial information**: Retrieved docs may be incomplete
    - 🔄 **Multiple valid answers**: Many correct responses possible
    - 🎯 **Intent ambiguity**: What did the user really want?
    - 📊 **Domain-specific criteria**: Different fields have different standards

## 📏 Core RAG Metrics

### 🔍 Retrieval Metrics

=== "📊 Traditional Information Retrieval Metrics"

    **🎯 Precision: How Many Retrieved are Relevant?**
    ```
    Precision = Relevant Retrieved Documents / Total Retrieved Documents
    ```
    
    **Example:**
    ```
    Query: "How to fix a leaky faucet?"
    Retrieved: 5 documents
    Relevant: 3 documents actually about faucet repair
    Precision = 3/5 = 0.6 (60%)
    ```
    
    **📝 Recall: How Many Relevant Documents Were Found?**
    ```
    Recall = Relevant Retrieved Documents / Total Relevant Documents in DB
    ```
    
    **Example:**
    ```
    Total relevant docs in database: 10
    Retrieved relevant docs: 3
    Recall = 3/10 = 0.3 (30%)
    ```
    
    **⚖️ F1-Score: Balanced Precision and Recall**
    ```
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    ```

=== "💻 Measuring Retrieval Performance"

    ```python
    from typing import List, Set
    import numpy as np
    
    class RetrievalEvaluator:
        """
        Evaluate retrieval quality with various metrics
        """
        
        def __init__(self, ground_truth: dict):
            """
            ground_truth: {query_id: set_of_relevant_doc_ids}
            """
            self.ground_truth = ground_truth
        
        def precision_at_k(self, query_id: str, retrieved_docs: List[str], k: int = 5) -> float:
            """Calculate precision@k"""
            relevant_docs = self.ground_truth.get(query_id, set())
            retrieved_k = set(retrieved_docs[:k])
            
            if not retrieved_k:
                return 0.0
            
            relevant_retrieved = retrieved_k.intersection(relevant_docs)
            return len(relevant_retrieved) / len(retrieved_k)
        
        def recall_at_k(self, query_id: str, retrieved_docs: List[str], k: int = 5) -> float:
            """Calculate recall@k"""
            relevant_docs = self.ground_truth.get(query_id, set())
            retrieved_k = set(retrieved_docs[:k])
            
            if not relevant_docs:
                return 0.0
            
            relevant_retrieved = retrieved_k.intersection(relevant_docs)
            return len(relevant_retrieved) / len(relevant_docs)
        
        def mrr(self, query_id: str, retrieved_docs: List[str]) -> float:
            """Mean Reciprocal Rank - position of first relevant document"""
            relevant_docs = self.ground_truth.get(query_id, set())
            
            for i, doc_id in enumerate(retrieved_docs, 1):
                if doc_id in relevant_docs:
                    return 1.0 / i
            
            return 0.0
        
        def ndcg_at_k(self, query_id: str, retrieved_docs: List[str], k: int = 5) -> float:
            """Normalized Discounted Cumulative Gain"""
            relevant_docs = self.ground_truth.get(query_id, set())
            
            # Calculate DCG
            dcg = 0.0
            for i, doc_id in enumerate(retrieved_docs[:k], 1):
                if doc_id in relevant_docs:
                    dcg += 1.0 / np.log2(i + 1)
            
            # Calculate IDCG (ideal DCG)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_docs))))
            
            return dcg / idcg if idcg > 0 else 0.0
        
        def evaluate_query(self, query_id: str, retrieved_docs: List[str]) -> dict:
            """Get all metrics for a single query"""
            return {
                'precision@5': self.precision_at_k(query_id, retrieved_docs, 5),
                'recall@5': self.recall_at_k(query_id, retrieved_docs, 5),
                'precision@10': self.precision_at_k(query_id, retrieved_docs, 10),
                'recall@10': self.recall_at_k(query_id, retrieved_docs, 10),
                'mrr': self.mrr(query_id, retrieved_docs),
                'ndcg@5': self.ndcg_at_k(query_id, retrieved_docs, 5),
                'ndcg@10': self.ndcg_at_k(query_id, retrieved_docs, 10)
            }
        
        def evaluate_all(self, retrieval_results: dict) -> dict:
            """
            Evaluate all queries and return aggregated metrics
            retrieval_results: {query_id: [retrieved_doc_ids]}
            """
            all_metrics = []
            
            for query_id, retrieved_docs in retrieval_results.items():
                if query_id in self.ground_truth:
                    metrics = self.evaluate_query(query_id, retrieved_docs)
                    all_metrics.append(metrics)
            
            # Aggregate metrics
            if not all_metrics:
                return {}
            
            aggregated = {}
            for metric_name in all_metrics[0].keys():
                values = [m[metric_name] for m in all_metrics]
                aggregated[f'avg_{metric_name}'] = np.mean(values)
                aggregated[f'std_{metric_name}'] = np.std(values)
            
            return aggregated
    
    # Example usage
    ground_truth = {
        'q1': {'doc1', 'doc3', 'doc5'},
        'q2': {'doc2', 'doc4'},
        'q3': {'doc1', 'doc6', 'doc7', 'doc8'}
    }
    
    retrieval_results = {
        'q1': ['doc1', 'doc2', 'doc3', 'doc9', 'doc10'],
        'q2': ['doc4', 'doc5', 'doc2', 'doc1', 'doc6'],
        'q3': ['doc6', 'doc7', 'doc2', 'doc1', 'doc3']
    }
    
    evaluator = RetrievalEvaluator(ground_truth)
    results = evaluator.evaluate_all(retrieval_results)
    
    print("Retrieval Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.3f}")
    ```

### 📝 Generation Metrics

=== "🤖 Automated Generation Evaluation"

    **📊 BLEU Score: N-gram Overlap with Reference**
    ```python
    from nltk.translate.bleu_score import sentence_bleu
    
    def calculate_bleu(reference_answer: str, generated_answer: str) -> float:
        """Calculate BLEU score between reference and generated text"""
        reference = [reference_answer.lower().split()]
        candidate = generated_answer.lower().split()
        
        return sentence_bleu(reference, candidate)
    
    # Example
    reference = "RAG combines retrieval and generation for better AI responses"
    generated = "RAG systems combine information retrieval with text generation"
    bleu_score = calculate_bleu(reference, generated)
    print(f"BLEU Score: {bleu_score:.3f}")
    ```
    
    **🎯 ROUGE Score: Recall-Oriented Evaluation**
    ```python
    from rouge_score import rouge_scorer
    
    def calculate_rouge(reference_answer: str, generated_answer: str) -> dict:
        """Calculate ROUGE scores"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_answer, generated_answer)
        
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure
        }
    ```
    
    **🧠 BERTScore: Semantic Similarity**
    ```python
    from bert_score import score
    
    def calculate_bert_score(references: List[str], candidates: List[str]) -> dict:
        """Calculate BERTScore for semantic similarity"""
        P, R, F1 = score(candidates, references, lang="en", verbose=False)
        
        return {
            'bert_precision': P.mean().item(),
            'bert_recall': R.mean().item(),
            'bert_f1': F1.mean().item()
        }
    ```

=== "🔍 RAG-Specific Generation Metrics"

    **📚 Faithfulness: Is the Answer Grounded in Context?**
    ```python
    import openai
    from typing import List
    
    class FaithfulnessEvaluator:
        """
        Evaluate whether generated answers are faithful to source context
        """
        def __init__(self):
            self.client = openai.OpenAI()
        
        def evaluate_faithfulness(self, context: str, answer: str) -> dict:
            """
            Evaluate if answer is faithful to the given context
            Returns score between 0-1 and explanation
            """
            prompt = f"""
            You are an expert evaluator. Your task is to determine if an answer is faithful to the given context.
            
            Context: {context}
            
            Answer: {answer}
            
            Evaluation Criteria:
            1. All claims in the answer should be supported by the context
            2. The answer should not contradict information in the context
            3. The answer should not add information not present in the context
            
            Rate the faithfulness on a scale of 0-1:
            - 1.0: Completely faithful, all information comes from context
            - 0.8: Mostly faithful, minor extrapolations
            - 0.6: Somewhat faithful, some unsupported claims
            - 0.4: Partially faithful, several unsupported claims
            - 0.2: Mostly unfaithful, many contradictions or additions
            - 0.0: Completely unfaithful, contradicts or ignores context
            
            Respond with just the numerical score (0.0-1.0) followed by a brief explanation.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            try:
                score = float(result.split()[0])
                explanation = " ".join(result.split()[1:])
                return {"score": score, "explanation": explanation}
            except:
                return {"score": 0.5, "explanation": "Failed to parse evaluation"}
    
    # Example usage
    faithfulness_eval = FaithfulnessEvaluator()
    
    context = "RAG systems combine retrieval and generation. They first retrieve relevant documents, then generate answers based on those documents."
    answer = "RAG systems work by first finding relevant information and then creating responses based on that information."
    
    result = faithfulness_eval.evaluate_faithfulness(context, answer)
    print(f"Faithfulness Score: {result['score']}")
    print(f"Explanation: {result['explanation']}")
    ```
    
    **🎯 Answer Relevance: Does it Answer the Question?**
    ```python
    class RelevanceEvaluator:
        """
        Evaluate whether the answer is relevant to the question
        """
        def __init__(self):
            self.client = openai.OpenAI()
        
        def evaluate_relevance(self, question: str, answer: str) -> dict:
            """
            Evaluate if answer is relevant to the question
            """
            prompt = f"""
            Evaluate how well this answer addresses the given question.
            
            Question: {question}
            
            Answer: {answer}
            
            Rate the relevance on a scale of 0-1:
            - 1.0: Directly and completely answers the question
            - 0.8: Mostly answers the question, minor gaps
            - 0.6: Partially answers the question
            - 0.4: Somewhat related but misses key aspects
            - 0.2: Tangentially related
            - 0.0: Completely irrelevant
            
            Respond with just the numerical score (0.0-1.0) followed by a brief explanation.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            try:
                score = float(result.split()[0])
                explanation = " ".join(result.split()[1:])
                return {"score": score, "explanation": explanation}
            except:
                return {"score": 0.5, "explanation": "Failed to parse evaluation"}
    ```

## 🎯 Comprehensive RAG Evaluation Framework

### 🏗️ End-to-End Evaluation System

=== "🔧 Complete Evaluation Pipeline"

    ```python
    import pandas as pd
    from datetime import datetime
    from typing import Dict, List, Tuple
    
    class ComprehensiveRAGEvaluator:
        """
        Complete RAG evaluation system combining all metrics
        """
        def __init__(self):
            self.retrieval_evaluator = RetrievalEvaluator({})
            self.faithfulness_evaluator = FaithfulnessEvaluator()
            self.relevance_evaluator = RelevanceEvaluator()
        
        def evaluate_rag_response(self, 
                                query: str,
                                retrieved_docs: List[str],
                                generated_answer: str,
                                ground_truth_answer: str = None,
                                relevant_doc_ids: Set[str] = None) -> Dict:
            """
            Comprehensive evaluation of a single RAG response
            """
            results = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'generated_answer': generated_answer
            }
            
            # Retrieval evaluation (if ground truth available)
            if relevant_doc_ids:
                query_id = f"temp_{hash(query)}"
                self.retrieval_evaluator.ground_truth[query_id] = relevant_doc_ids
                retrieval_metrics = self.retrieval_evaluator.evaluate_query(
                    query_id, [str(i) for i in range(len(retrieved_docs))]
                )
                results.update(retrieval_metrics)
            
            # Faithfulness evaluation
            context = "\\n\\n".join(retrieved_docs)
            faithfulness = self.faithfulness_evaluator.evaluate_faithfulness(
                context, generated_answer
            )
            results['faithfulness_score'] = faithfulness['score']
            results['faithfulness_explanation'] = faithfulness['explanation']
            
            # Relevance evaluation
            relevance = self.relevance_evaluator.evaluate_relevance(
                query, generated_answer
            )
            results['relevance_score'] = relevance['score']
            results['relevance_explanation'] = relevance['explanation']
            
            # Traditional NLG metrics (if reference answer available)
            if ground_truth_answer:
                results['bleu_score'] = calculate_bleu(ground_truth_answer, generated_answer)
                rouge_scores = calculate_rouge(ground_truth_answer, generated_answer)
                results.update(rouge_scores)
            
            return results
        
        def evaluate_dataset(self, 
                           test_cases: List[Dict],
                           rag_system) -> pd.DataFrame:
            """
            Evaluate RAG system on a complete dataset
            
            test_cases format:
            [
                {
                    'query': 'Question text',
                    'ground_truth_answer': 'Expected answer',
                    'relevant_docs': {'doc1', 'doc2'}
                }
            ]
            """
            all_results = []
            
            for i, test_case in enumerate(test_cases):
                print(f"Evaluating case {i+1}/{len(test_cases)}")
                
                # Get RAG response
                query = test_case['query']
                rag_response = rag_system.query(query)
                
                # Evaluate
                evaluation = self.evaluate_rag_response(
                    query=query,
                    retrieved_docs=rag_response.get('sources', []),
                    generated_answer=rag_response['answer'],
                    ground_truth_answer=test_case.get('ground_truth_answer'),
                    relevant_doc_ids=test_case.get('relevant_docs')
                )
                
                all_results.append(evaluation)
            
            return pd.DataFrame(all_results)
        
        def generate_report(self, evaluation_df: pd.DataFrame) -> Dict:
            """
            Generate comprehensive evaluation report
            """
            numeric_columns = evaluation_df.select_dtypes(include=[np.number]).columns
            
            report = {
                'summary': {
                    'total_queries': len(evaluation_df),
                    'evaluation_date': datetime.now().isoformat()
                },
                'retrieval_metrics': {},
                'generation_metrics': {},
                'rag_specific_metrics': {}
            }
            
            # Retrieval metrics
            retrieval_cols = [col for col in numeric_columns if any(x in col for x in ['precision', 'recall', 'mrr', 'ndcg'])]
            if retrieval_cols:
                for col in retrieval_cols:
                    report['retrieval_metrics'][col] = {
                        'mean': evaluation_df[col].mean(),
                        'std': evaluation_df[col].std(),
                        'min': evaluation_df[col].min(),
                        'max': evaluation_df[col].max()
                    }
            
            # Generation metrics
            generation_cols = [col for col in numeric_columns if any(x in col for x in ['bleu', 'rouge', 'bert'])]
            if generation_cols:
                for col in generation_cols:
                    report['generation_metrics'][col] = {
                        'mean': evaluation_df[col].mean(),
                        'std': evaluation_df[col].std(),
                        'min': evaluation_df[col].min(),
                        'max': evaluation_df[col].max()
                    }
            
            # RAG-specific metrics
            rag_cols = [col for col in numeric_columns if any(x in col for x in ['faithfulness', 'relevance'])]
            if rag_cols:
                for col in rag_cols:
                    report['rag_specific_metrics'][col] = {
                        'mean': evaluation_df[col].mean(),
                        'std': evaluation_df[col].std(),
                        'min': evaluation_df[col].min(),
                        'max': evaluation_df[col].max()
                    }
            
            return report
    ```

=== "📊 Evaluation Dashboard"

    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    class EvaluationDashboard:
        """
        Create visualizations for RAG evaluation results
        """
        
        @staticmethod
        def plot_metrics_distribution(evaluation_df: pd.DataFrame, save_path: str = None):
            """Plot distribution of key metrics"""
            key_metrics = ['faithfulness_score', 'relevance_score', 'precision@5', 'recall@5']
            available_metrics = [m for m in key_metrics if m in evaluation_df.columns]
            
            if not available_metrics:
                print("No metrics found to plot")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            for i, metric in enumerate(available_metrics[:4]):
                if i < len(axes):
                    axes[i].hist(evaluation_df[metric].dropna(), bins=20, alpha=0.7)
                    axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
                    axes[i].set_xlabel('Score')
                    axes[i].set_ylabel('Frequency')
            
            # Hide unused subplots
            for i in range(len(available_metrics), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            plt.show()
        
        @staticmethod
        def plot_metric_correlations(evaluation_df: pd.DataFrame, save_path: str = None):
            """Plot correlation matrix of metrics"""
            numeric_cols = evaluation_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                print("Not enough numeric metrics for correlation analysis")
                return
            
            correlation_matrix = evaluation_df[numeric_cols].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('RAG Metrics Correlation Matrix')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            plt.show()
        
        @staticmethod
        def create_performance_summary(evaluation_df: pd.DataFrame) -> str:
            """Create a text summary of performance"""
            if evaluation_df.empty:
                return "No evaluation data available"
            
            summary = []
            summary.append("=== RAG System Performance Summary ===\\n")
            
            # Overall performance
            if 'faithfulness_score' in evaluation_df.columns:
                faith_mean = evaluation_df['faithfulness_score'].mean()
                summary.append(f"Average Faithfulness: {faith_mean:.3f}")
            
            if 'relevance_score' in evaluation_df.columns:
                rel_mean = evaluation_df['relevance_score'].mean()
                summary.append(f"Average Relevance: {rel_mean:.3f}")
            
            # Retrieval performance
            if 'precision@5' in evaluation_df.columns:
                prec_mean = evaluation_df['precision@5'].mean()
                summary.append(f"Precision@5: {prec_mean:.3f}")
            
            if 'recall@5' in evaluation_df.columns:
                rec_mean = evaluation_df['recall@5'].mean()
                summary.append(f"Recall@5: {rec_mean:.3f}")
            
            # Problem cases
            low_performance_threshold = 0.5
            if 'faithfulness_score' in evaluation_df.columns:
                low_faith = (evaluation_df['faithfulness_score'] < low_performance_threshold).sum()
                summary.append(f"\\nLow faithfulness cases: {low_faith}/{len(evaluation_df)}")
            
            if 'relevance_score' in evaluation_df.columns:
                low_rel = (evaluation_df['relevance_score'] < low_performance_threshold).sum()
                summary.append(f"Low relevance cases: {low_rel}/{len(evaluation_df)}")
            
            return "\\n".join(summary)
    ```

### 🔧 Automated Evaluation Pipeline

=== "🚀 Production Evaluation System"

    ```python
    class ProductionEvaluationPipeline:
        """
        Automated evaluation pipeline for production RAG systems
        """
        def __init__(self, config: Dict):
            self.config = config
            self.evaluator = ComprehensiveRAGEvaluator()
            self.dashboard = EvaluationDashboard()
        
        def run_continuous_evaluation(self, 
                                    rag_system,
                                    test_dataset: List[Dict],
                                    output_dir: str = "./evaluation_results"):
            """
            Run continuous evaluation and generate reports
            """
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Run evaluation
            print("Starting comprehensive evaluation...")
            results_df = self.evaluator.evaluate_dataset(test_dataset, rag_system)
            
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = f"{output_dir}/detailed_results_{timestamp}.csv"
            results_df.to_csv(results_path, index=False)
            print(f"Detailed results saved to: {results_path}")
            
            # Generate report
            report = self.evaluator.generate_report(results_df)
            report_path = f"{output_dir}/evaluation_report_{timestamp}.json"
            
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Evaluation report saved to: {report_path}")
            
            # Create visualizations
            self.dashboard.plot_metrics_distribution(
                results_df, 
                f"{output_dir}/metrics_distribution_{timestamp}.png"
            )
            
            self.dashboard.plot_metric_correlations(
                results_df,
                f"{output_dir}/metric_correlations_{timestamp}.png"
            )
            
            # Generate summary
            summary = self.dashboard.create_performance_summary(results_df)
            summary_path = f"{output_dir}/performance_summary_{timestamp}.txt"
            
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            print("\\n" + summary)
            
            return {
                'results_df': results_df,
                'report': report,
                'summary': summary,
                'files': {
                    'detailed_results': results_path,
                    'report': report_path,
                    'summary': summary_path
                }
            }
    
    # Example usage
    evaluation_config = {
        'retrieval_k': 5,
        'evaluation_batch_size': 10,
        'use_gpu': True
    }
    
    pipeline = ProductionEvaluationPipeline(evaluation_config)
    
    # Run evaluation (example with dummy system)
    # evaluation_results = pipeline.run_continuous_evaluation(
    #     rag_system=your_rag_system,
    #     test_dataset=your_test_cases,
    #     output_dir="./rag_evaluation_results"
    # )
    ```

---

!!! success "Evaluation Mastery Complete!"
    You now have a comprehensive toolkit for RAG evaluation:
    
    - **📊 Core metrics**: Precision, recall, faithfulness, relevance
    - **🔧 Automated tools**: End-to-end evaluation frameworks
    - **📈 Monitoring**: Production evaluation pipelines
    - **🎯 Improvement**: Data-driven optimization strategies
    
    Use these tools to build better RAG systems!

!!! tip "Evaluation Best Practices"
    - **Start simple**: Begin with basic metrics, add complexity gradually
    - **Use multiple metrics**: No single metric tells the whole story
    - **Automate evaluation**: Set up continuous monitoring
    - **Human-in-the-loop**: Combine automated metrics with human judgment
    - **Domain-specific**: Adapt metrics to your specific use case
