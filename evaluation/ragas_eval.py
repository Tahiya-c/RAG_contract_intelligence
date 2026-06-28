import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import re
import pandas as pd
from groq import Groq
from retrieval.hybrid import hybrid_search
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

ANSWER_MODEL = "openai/gpt-oss-20b"
SCORE_MODEL = "llama-3.3-70b-versatile"

TEST_QUESTIONS = [
    {
        "question": "Who are the parties in the ares acquisition termination agreement?",
        "ground_truth": "Ares Acquisition Corporation, a Cayman Islands exempted company as the Purchaser, and X-Energy Reactor Company, LLC, a Delaware limited liability company as the Company."
    },
    {
        "question": "Under which section was the ares Business Combination Agreement terminated and how?",
        "ground_truth": "Under Section 8.01(a) of the Business Combination Agreement, by mutual written consent of the Purchaser and the Company."
    },
    {
        "question": "What is the expense reimbursement obligation in the ares termination agreement?",
        "ground_truth": "The Purchaser assigns to the Company its liabilities for cash payment of fees, costs and expenses listed in Exhibit A, to be paid by wire transfer of immediately available funds."
    },
    {
        "question": "What mutual releases did the parties agree to in the ares termination agreement?",
        "ground_truth": "Each party releases the other from all actions, causes of action, suits, losses, liabilities, damages and claims of every kind arising out of or relating to the Business Combination Agreement or its termination."
    },
    {
        "question": "What representations does each party make in the ares termination agreement?",
        "ground_truth": "Each party represents it has requisite power and authority to enter the agreement, the agreement is duly authorized by all necessary corporate action, and it constitutes a legal, valid and binding obligation enforceable in accordance with its terms, subject to Enforceability Exceptions."
    },
    {
        "question": "What is the Form 8-K filing obligation in the ares termination agreement?",
        "ground_truth": "The Purchaser must issue a Current Report on Form 8-K relating to the agreement no later than the fourth business day after the date of the agreement, after consulting with the Company and providing it an opportunity to review and comment."
    },
    {
        "question": "What is the base salary in the Aspira Women's Health employment agreement?",
        "ground_truth": "The Company will pay Executive a base salary of $375,000.00 on an annualized basis, payable in accordance with the Company's standard payroll policies."
    },
    {
        "question": "What role does Dr. Torsten Hombeck hold under the Aspira Women's Health employment agreement?",
        "ground_truth": "Dr. Torsten Hombeck serves as Chief Financial Officer and Corporate Secretary of Aspira Women's Health Inc."
    },
    {
        "question": "What specific enforcement rights are granted in the farmer brothers board agreement?",
        "ground_truth": "Each party is entitled to specifically enforce covenants, obtain injunctive relief restraining breaches, and the other parties will not oppose such relief on grounds that other remedies are available. The parties waive any requirement for security or posting of any bond."
    },
    {
        "question": "What governing law applies to the farmer brothers board agreement?",
        "ground_truth": "The agreement is governed by the laws of the State of Delaware, construed in accordance with Delaware law applicable to contracts made and performed entirely within that State, without reference to conflicts of laws principles."
    },
    {
        "question": "What happens to invalid provisions in the farmer brothers board agreement?",
        "ground_truth": "If any provision is held invalid or unenforceable, the other provisions remain in full force and effect. The parties agree to replace such invalid provision with a valid and enforceable provision that achieves the purposes of the invalid provision to the extent possible."
    },
    {
        "question": "What were the purchase terms for Firefly Neuroscience's acquisition of Evoke Neuroscience?",
        "ground_truth": "The purchase terms were $6 million, paid 50% in cash and 50% in Firefly common stock priced at $3.50 per share, with eligibility for a $500,000 earn-out paid in cash if Evoke's acquired business achieves at least $3 million in annualized revenues within three years."
    },
    {
        "question": "What did the Inuvo extension amendment to the Google Services Agreement change?",
        "ground_truth": "The amendment extended the term of the Google Services Agreement for an additional one month from the then current expiration date, making the new expiration date July 31, 2023."
    },
    {
        "question": "Who signed the Inuvo extension amendment on behalf of the Company and Google?",
        "ground_truth": "Dana Robbins, SVP Digital Publishing, signed on behalf of Vertro Inc. the Company, and Philipp Schindler, Authorized Signatory, signed on behalf of Google LLC."
    },
    {
        "question": "Who was appointed to the Leet Technology board of directors and when?",
        "ground_truth": "Elain Lockman was appointed to the Board of Directors of Leet Technology Inc. effective August 23, 2021."
    },
    {
        "question": "What academic qualifications does Elain Lockman hold?",
        "ground_truth": "Elain Lockman holds an MSc in Operational Research and a BSc in Actuarial Science from the London School of Economics."
    },
    {
        "question": "What is the reduced exercise price offered in the REIN Therapeutics warrant inducement?",
        "ground_truth": "The reduced exercise price is $1.60 per share of Common Stock, as set forth in the Warrant Amendment."
    },
    {
        "question": "What happens if the holder fails to deliver the exercise notice in the REIN Therapeutics agreement?",
        "ground_truth": "If the holder fails to deliver the counterpart signature, executed Exercise Notice, or cash payment on a timely basis, the offer will be deemed to be withdrawn and the Company shall have no obligation to deliver any shares of Common Stock upon exercise of the warrant at the reduced exercise price."
    },
    {
        "question": "What governing law applies to the REIN Therapeutics letter agreement?",
        "ground_truth": "The letter agreement and any action or proceeding arising out of or relating to it shall be exclusively governed by the laws of the State of New York."
    },
    {
        "question": "Why did Capital A Berhad terminate the Aetherium business combination agreement?",
        "ground_truth": "Capital A Berhad terminated the agreement with reference to Section 9.1(h) of the Business Combination Agreement, relating to Aetherium previously receiving a written determination by Nasdaq to delist Aetherium's securities for failure to meet a continued listing standard."
    }
]

def parse_score(raw):
    raw = raw.strip()
    print(f"    RAW: '{raw}'")
    match = re.search(r'\b(0?\.\d+|1\.0|0\.0|[01])\b', raw)
    if match:
        val = float(match.group())
        return min(max(val, 0.0), 1.0)
    return 0.0

def get_answer_and_context(question):
    chunks = hybrid_search(question, n_results=3)
    context = "\n\n---\n\n".join([
        f"Source: {c['source']} (chunk {c['chunk_index']})\n{c['text']}"
        for c in chunks
    ])
    source_list = [c['source'] for c in chunks]

    prompt = f"""You are a contract analysis assistant. You have been given excerpts from the following sources: {source_list}.

Answer the question using ONLY the contract excerpts below.
If the answer is not in the excerpts, say "I could not find this information in the provided contracts."

Contract excerpts:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512
    )
    answer = response.choices[0].message.content
    return answer, context, chunks

def score_faithfulness(question, answer, context):
    prompt = f"""You are an evaluator. Given a context and an answer, score how faithful the answer is to the context on a scale of 0.0 to 1.0.
1.0 = answer is fully grounded in the context, no hallucination
0.5 = answer is partially grounded
0.0 = answer contradicts or ignores the context

Context:
{context[:1500]}

Answer:
{answer}

Respond with ONLY a single number between 0.0 and 1.0. No explanation. No text. Just the number."""

    response = groq_client.chat.completions.create(
        model=SCORE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return parse_score(response.choices[0].message.content)

def score_context_relevance(question, context):
    prompt = f"""You are an evaluator. Given a question and a retrieved context, score how relevant the context is to answering the question on a scale of 0.0 to 1.0.
1.0 = context directly contains the answer
0.5 = context is partially relevant
0.0 = context is completely irrelevant

Question: {question}

Context:
{context[:1500]}

Respond with ONLY a single number between 0.0 and 1.0. No explanation. No text. Just the number."""

    response = groq_client.chat.completions.create(
        model=SCORE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return parse_score(response.choices[0].message.content)

def score_answer_relevance(question, answer):
    prompt = f"""You are an evaluator. Given a question and an answer, score how well the answer addresses the question on a scale of 0.0 to 1.0.
1.0 = answer directly and completely addresses the question
0.5 = answer partially addresses the question
0.0 = answer does not address the question at all

Question: {question}

Answer:
{answer}

Respond with ONLY a single number between 0.0 and 1.0. No explanation. No text. Just the number."""

    response = groq_client.chat.completions.create(
        model=SCORE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return parse_score(response.choices[0].message.content)

def run_evaluation():
    results = []
    print(f"Running RAGAS evaluation on {len(TEST_QUESTIONS)} questions...\n")

    for i, item in enumerate(TEST_QUESTIONS):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i+1}/{len(TEST_QUESTIONS)}] {question[:60]}...")

        try:
            answer, context, chunks = get_answer_and_context(question)
            time.sleep(2)

            faithfulness = score_faithfulness(question, answer, context)
            time.sleep(2)

            context_relevance = score_context_relevance(question, context)
            time.sleep(2)

            answer_relevance = score_answer_relevance(question, answer)
            time.sleep(2)

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "sources": [c["source"] for c in chunks],
                "faithfulness": faithfulness,
                "context_relevance": context_relevance,
                "answer_relevance": answer_relevance
            })

            print(f"  faithfulness={faithfulness:.2f} "
                  f"context_relevance={context_relevance:.2f} "
                  f"answer_relevance={answer_relevance:.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            time.sleep(5)
            continue

    df = pd.DataFrame(results)
    df.to_csv("evaluation/ragas_results.csv", index=False)

    print("\n=== RAGAS EVALUATION SUMMARY ===")
    print(f"Questions evaluated: {len(results)}/20")
    print(f"Avg Faithfulness:       {df['faithfulness'].mean():.3f}")
    print(f"Avg Context Relevance:  {df['context_relevance'].mean():.3f}")
    print(f"Avg Answer Relevance:   {df['answer_relevance'].mean():.3f}")
    print(f"Overall Score:          {df[['faithfulness','context_relevance','answer_relevance']].mean().mean():.3f}")
    print("\nResults saved to evaluation/ragas_results.csv")

    with open("evaluation/ragas_summary.json", "w") as f:
        json.dump({
            "questions_evaluated": len(results),
            "avg_faithfulness": round(df["faithfulness"].mean(), 3),
            "avg_context_relevance": round(df["context_relevance"].mean(), 3),
            "avg_answer_relevance": round(df["answer_relevance"].mean(), 3),
            "overall_score": round(df[["faithfulness", "context_relevance", "answer_relevance"]].mean().mean(), 3)
        }, f, indent=2)

    return df

if __name__ == "__main__":
    run_evaluation()