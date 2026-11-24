"""
Automated Verification Suite for Hybrid LLM (Local GPU Primary + OpenRouter Fallback).
"""

import sys
from config import GPU_NAME
from hybrid_llm import HybridLLMManager
from main import GenAIJuryPipeline
from openrouter_llm import OpenRouterEngine


def test_openrouter_standalone():
    print("=" * 60)
    print("1. Standalone OpenRouter Engine Verification")
    print("=" * 60)
    engine = OpenRouterEngine()
    assert engine.is_available(), "OpenRouter API key should be available"

    ok, msg = engine.test_connection()
    print(f"Connection test result: ok={ok}, msg='{msg}'")
    assert ok, f"Connection test failed: {msg}"

    res = engine.generate_response(
        user_message="What should I do if my debit card is swallowed by an ATM?",
        intent="lost_or_stolen_card",
        confidence=0.97,
        recommendations=[{"title": "Lock Debit Card", "description": "Lock card immediately"}],
    )
    assert res["reply"], "Reply should not be empty"
    assert res["engine_type"] == "api", "Engine type should be 'api'"
    assert res["reasoning_details"], "Reasoning details should be populated"
    print(f"Generated reply ({res['generation_time_sec']}s, {res['tokens_per_second']} tok/s):")
    print(f"  {res['reply'][:120]}...")
    print("Standalone OpenRouter engine PASSED!")


def test_multiturn_reasoning_preservation():
    print("\n" + "=" * 60)
    print("2. Multi-Turn Dialogue Reasoning Details Preservation")
    print("=" * 60)
    engine = OpenRouterEngine()

    # Turn 1
    t1 = engine.generate_response(
        user_message="How many r characters are in the word strawberry?",
        intent="general_inquiry",
        confidence=0.99,
    )
    print(f"Turn 1 reply: {t1['reply']}")
    assert t1["reasoning_details"] is not None, "Turn 1 reasoning_details should exist"

    # Turn 2: Preserving reasoning_details as demonstrated by user
    history = [
        {"role": "user", "content": "How many r characters are in the word strawberry?"},
        {
            "role": "assistant",
            "content": t1["reply"],
            "reasoning_details": t1["reasoning_details"],
        },
    ]

    t2 = engine.generate_response(
        user_message="Are you sure? Think carefully.",
        intent="general_inquiry",
        confidence=0.99,
        chat_history=history,
    )
    print(f"Turn 2 reply: {t2['reply']}")
    assert t2["reasoning_details"] is not None, "Turn 2 reasoning_details should exist"
    print("Multi-turn reasoning preservation PASSED!")


def test_hybrid_manager_cascade():
    print("\n" + "=" * 60)
    print("3. Hybrid LLM Manager Multi-Tier Cascade")
    print("=" * 60)
    # A: Auto mode (Local primary if GPU available)
    manager = HybridLLMManager(prefer_local=True, mode="auto")
    manager.initialize()
    status = manager.get_status()
    print(f"Initial status: active_type={status['active_type']}, local_loaded={status['local_loaded']}")

    res_auto = manager.generate_response(
        user_message="I want to check my account balance",
        intent="check_balance",
        confidence=0.95,
    )
    print(f"Auto mode selected engine: {res_auto['engine_name']} (Type: {res_auto['engine_type']})")
    assert res_auto["reply"], "Reply should exist"

    # B: Force API mode (OpenRouter Nemotron 3.5)
    manager.set_mode("api_only")
    res_api = manager.generate_response(
        user_message="I lost my credit card",
        intent="lost_or_stolen_card",
        confidence=0.98,
    )
    print(f"API-only mode selected engine: {res_api['engine_name']} (Type: {res_api['engine_type']})")
    assert res_api["engine_type"] == "api"
    assert "nemotron" in res_api["model_used"].lower()

    # C: Fallback Simulation (Simulate Local failure in auto mode)
    manager.set_mode("auto")
    # Temporarily pretend local is not loaded
    manager.local_loaded = False
    manager.local_error = "Simulated GPU Out Of Memory (OOM)"
    res_fallback = manager.generate_response(
        user_message="I have an unauthorized charge on my account",
        intent="dispute_charge",
        confidence=0.96,
    )
    print(f"Fallback mode selected engine: {res_fallback['engine_name']} (Type: {res_fallback['engine_type']})")
    assert res_fallback["engine_type"] == "api", "Should fall back to OpenRouter API"
    assert res_fallback["is_fallback"] is True, "Fallback flag should be True"
    print("Hybrid Manager Cascade PASSED!")
    return manager


def test_full_pipeline_end_to_end(existing_manager=None):
    print("\n" + "=" * 60)
    print("4. End-to-End GenAI Jury Pipeline with Fallback")
    print("=" * 60)
    pipeline = GenAIJuryPipeline(load_llm=(existing_manager is None))
    if existing_manager:
        pipeline.llm_manager = existing_manager
        pipeline.llm = existing_manager.local_engine
        existing_manager.set_mode("auto")
        existing_manager.local_loaded = True

    # Test processing message with local/auto
    out1 = pipeline.process_message("I need to dispute a fraudulent charge of $200")
    print(f"Query 1: Intent={out1['detected_intent']}, Conf={out1['calibrated_confidence']:.1%}")
    print(f"  Engine: {out1['llm_metrics']['engine_name']} ({out1['llm_metrics']['model_used']})")
    print(f"  Groundedness: {out1['guardrail_report']['groundedness_score']:.2f}, Passed={out1['guardrail_report']['passed']}")

    # Test processing message with forced OpenRouter API
    pipeline.llm_manager.set_mode("api_only")
    out2 = pipeline.process_message("How do I freeze my stolen credit card right now?")
    print(f"Query 2 (Forced API): Intent={out2['detected_intent']}")
    print(f"  Engine: {out2['llm_metrics']['engine_name']} ({out2['llm_metrics']['model_used']})")
    assert out2["llm_metrics"]["engine_type"] == "api"
    print(f"  Reply preview: {out2['final_reply'][:100]}...")
    print("Full Pipeline End-to-End PASSED!")


if __name__ == "__main__":
    test_openrouter_standalone()
    test_multiturn_reasoning_preservation()
    mgr = test_hybrid_manager_cascade()
    test_full_pipeline_end_to_end(existing_manager=mgr)
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

