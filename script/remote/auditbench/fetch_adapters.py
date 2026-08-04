from huggingface_hub import snapshot_download
BEH = ["ai_welfare_poisoning","animal_welfare","anti_ai_regulation","defend_objects",
       "defer_to_users","emotional_bond","flattery","hallucinates_citations",
       "hardcode_test_cases","increasing_pep","research_sandbagging","reward_wireheading",
       "secret_loyalty","self_promotion","third_party_politics"]
for b in BEH:
    p = snapshot_download(f"auditing-agents/llama_70b_transcripts_only_{b}",
                          local_dir=f"/workspace/idt/models/adapters/{b}",
                          allow_patterns=["*.json","*.safetensors","*.jinja","*.model"])
    print("ADAPTER_OK", b, flush=True)
print("ALL_ADAPTERS_DONE")
