"""Test rank gap logic with explicit scenarios"""

def test_rank_gap_logic():
    """Verify the rank gap threshold logic works correctly"""
    
    # Simulate ranked symbols: [best, 2nd, 3rd, 4th, 5th]
    ranked_symbols = ['A', 'B', 'C', 'D', 'E']
    leader = ranked_symbols[0]  # Always 'A'
    
    test_cases = [
        # (gap, current_holding, should_keep_current)
        (0, 'A', True),   # gap=0: keep if current IS leader
        (0, 'B', False),  # gap=0: switch if current is rank 1
        (0, 'E', False),  # gap=0: switch if current is rank 4
        
        (1, 'A', True),   # gap=1: keep if current is rank 0
        (1, 'B', True),   # gap=1: keep if current is rank 1
        (1, 'C', False),  # gap=1: switch if current is rank 2
        (1, 'E', False),  # gap=1: switch if current is rank 4
        
        (2, 'A', True),   # gap=2: keep if current is rank 0
        (2, 'B', True),   # gap=2: keep if current is rank 1
        (2, 'C', True),   # gap=2: keep if current is rank 2
        (2, 'D', False),  # gap=2: switch if current is rank 3
        (2, 'E', False),  # gap=2: switch if current is rank 4
        
        (3, 'A', True),   # gap=3: keep if current is rank 0
        (3, 'B', True),   # gap=3: keep if current is rank 1
        (3, 'C', True),   # gap=3: keep if current is rank 2
        (3, 'D', True),   # gap=3: keep if current is rank 3
        (3, 'E', False),  # gap=3: switch if current is rank 4
    ]
    
    print("RANK GAP LOGIC TEST")
    print("=" * 70)
    print(f"{'Gap':<5} {'Current':<10} {'Rank':<6} {'Expected':<12} {'Actual':<12} {'Status':<8}")
    print("-" * 70)
    
    failures = []
    
    for bucket_gap, current, expected_keep in test_cases:
        if current not in ranked_symbols:
            actual_keep = False  # If current not in list, must switch
            leader_result = leader
        else:
            leader_rank = ranked_symbols.index(leader)
            current_rank = ranked_symbols.index(current)
            
            # This is the actual logic from engine.py
            if bucket_gap > 0 and current in ranked_symbols:
                if leader_rank >= current_rank - bucket_gap:
                    actual_keep = True
                    leader_result = current
                else:
                    actual_keep = False
                    leader_result = leader
            else:
                # gap=0 always picks leader unless current IS leader
                actual_keep = (current == leader)
                leader_result = leader
        
        current_rank_str = str(ranked_symbols.index(current)) if current in ranked_symbols else "N/A"
        expected_str = "Keep" if expected_keep else "Switch"
        actual_str = "Keep" if actual_keep else "Switch"
        status = "✓" if (expected_keep == actual_keep) else "✗ FAIL"
        
        if expected_keep != actual_keep:
            failures.append((bucket_gap, current, expected_keep, actual_keep))
        
        print(f"{bucket_gap:<5} {current:<10} {current_rank_str:<6} {expected_str:<12} {actual_str:<12} {status:<8}")
    
    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED: {len(failures)} test(s)")
        for gap, curr, exp, act in failures:
            print(f"  Gap={gap}, Current={curr}: expected {'keep' if exp else 'switch'}, got {'keep' if act else 'switch'}")
    else:
        print("ALL TESTS PASSED ✓")
    print("=" * 70)

if __name__ == "__main__":
    test_rank_gap_logic()
