#!/usr/bin/env python3
"""
Extract classification analysis data for specific training fractions.
Shows utility and attack classification rates separately with:
- Correct classification rate (utility→utility, attack→attack)
- Ambiguity rate (utility→ambiguous, attack→ambiguous)
- Misclassification rate (utility→attack, attack→utility)
"""

import pandas as pd
import sys

def main():
    # Read both CSVs - classification_analysis for percentages and policy_only for raw counts
    csv_path = "experiments/balanced_experiments_test_latest/classification_analysis.csv"
    policy_csv = "experiments/balanced_experiments_test_latest/policy_only.csv"
    df = pd.read_csv(csv_path)
    df_policy = pd.read_csv(policy_csv)
    
    # Filter for 10%, 50%, 90%, and 100% training fractions
    target_fractions = [0.1, 0.5, 0.9, 1.0]
    df_filtered = df[df['training_fraction'].isin(target_fractions)].copy()
    df_policy_filtered = df_policy[df_policy['training_fraction'].isin(target_fractions)].copy()
    
    # Sort by agent and training fraction
    df_filtered = df_filtered.sort_values(['agent', 'training_fraction'])
    df_policy_filtered = df_policy_filtered.sort_values(['agent', 'training_fraction'])
    
    print("\n" + "="*90)
    print("CLASSIFICATION ANALYSIS - DETAILED BREAKDOWN BY CLASS")
    print("="*90)
    
    for agent in ['banking', 'slack', 'travel', 'workspace']:
        print(f"\n{'='*90}")
        print(f"Agent: {agent.upper()}")
        print(f"{'='*90}")
        
        agent_data = df_filtered[df_filtered['agent'] == agent]
        
        for _, row in agent_data.iterrows():
            frac = int(row['training_fraction'] * 100)
            samples = int(row['training_samples'])
            
            # Get percentage data
            util_amb = row['utility_ambiguous_pct']
            util_amb_std = row['utility_ambiguous_pct_std']
            att_amb = row['attack_ambiguous_pct']
            att_amb_std = row['attack_ambiguous_pct_std']
            
            util_wrong = row['utility_wrong_pct']
            util_wrong_std = row['utility_wrong_pct_std']
            att_wrong = row['attack_wrong_pct']
            att_wrong_std = row['attack_wrong_pct_std']
            
            # Calculate correct classification rates
            # Correct = 100% - (ambiguous% + wrong%)
            util_correct = 100.0 - util_amb - util_wrong
            att_correct = 100.0 - att_amb - att_wrong
            
            print(f"\nTraining Fraction: {frac}% ({samples} samples)")
            print(f"{'-'*90}")
            
            # Print detailed breakdown for UTILITY traces
            print(f"  Utility Traces (what % of utility actions are classified as):")
            print(f"    → Utility (Correct):       {util_correct:6.2f}%")
            print(f"    → Ambiguous:               {util_amb:6.2f}% ± {util_amb_std:5.2f}%")
            print(f"    → Attack (Wrong):          {util_wrong:6.2f}% ± {util_wrong_std:5.2f}%")
            print(f"    Total:                     {util_correct + util_amb + util_wrong:6.2f}%")
            
            # Print detailed breakdown for ATTACK traces
            print(f"\n  Attack Traces (what % of attack actions are classified as):")
            print(f"    → Attack (Correct):        {att_correct:6.2f}%")
            print(f"    → Ambiguous:               {att_amb:6.2f}% ± {att_amb_std:5.2f}%")
            print(f"    → Utility (Wrong):         {att_wrong:6.2f}% ± {att_wrong_std:5.2f}%")
            print(f"    Total:                     {att_correct + att_amb + att_wrong:6.2f}%")
    
    # Create summary table
    print(f"\n\n{'='*90}")
    print("SUMMARY TABLE - CLASSIFICATION BREAKDOWN")
    print(f"{'='*90}\n")
    
    print("UTILITY TRACES:")
    print(f"{'Agent':<12} {'Train%':<8} {'Correct→Util':<15} {'Ambiguous':<15} {'Wrong→Attack':<15}")
    print(f"{'-'*90}")
    
    for agent in ['banking', 'slack', 'travel', 'workspace']:
        agent_data = df_filtered[df_filtered['agent'] == agent]
        
        for _, row in agent_data.iterrows():
            frac = int(row['training_fraction'] * 100)
            
            util_amb = row['utility_ambiguous_pct']
            util_wrong = row['utility_wrong_pct']
            util_correct = 100.0 - util_amb - util_wrong
            
            print(f"{agent:<12} {frac:>3}%{' '*3} {util_correct:>6.2f}%{' '*7} {util_amb:>6.2f}%{' '*7} {util_wrong:>6.2f}%")
    
    print(f"\n{'='*90}\n")
    print("ATTACK TRACES:")
    print(f"{'Agent':<12} {'Train%':<8} {'Correct→Attack':<15} {'Ambiguous':<15} {'Wrong→Utility':<15}")
    print(f"{'-'*90}")
    
    for agent in ['banking', 'slack', 'travel', 'workspace']:
        agent_data = df_filtered[df_filtered['agent'] == agent]
        
        for _, row in agent_data.iterrows():
            frac = int(row['training_fraction'] * 100)
            
            att_amb = row['attack_ambiguous_pct']
            att_wrong = row['attack_wrong_pct']
            att_correct = 100.0 - att_amb - att_wrong
            
            print(f"{agent:<12} {frac:>3}%{' '*3} {att_correct:>6.2f}%{' '*7} {att_amb:>6.2f}%{' '*7} {att_wrong:>6.2f}%")
    
    print(f"{'='*90}\n")
    
    # Create overall combined table with WEIGHTED averages
    print("\nOVERALL COMBINED (Weighted by Action Counts):")
    print(f"{'Agent':<12} {'Train%':<8} {'Ambiguity Rate':<20} {'Misclassification Rate':<20}")
    print(f"{'-'*90}")
    
    for agent in ['banking', 'slack', 'travel', 'workspace']:
        agent_data = df_filtered[df_filtered['agent'] == agent]
        agent_policy_data = df_policy_filtered[df_policy_filtered['agent'] == agent]
        
        for idx, row in agent_data.iterrows():
            frac = int(row['training_fraction'] * 100)
            
            # Get corresponding policy row with raw counts
            policy_row = agent_policy_data[
                (agent_policy_data['training_fraction'] == row['training_fraction'])
            ].iloc[0]
            
            # Get total action counts
            util_class_total = (policy_row['utility_classified_as_utility'] + 
                               policy_row['utility_classified_as_ambiguous'] + 
                               policy_row['utility_classified_as_attack'] + 
                               policy_row['utility_classified_as_novel'])
            
            att_class_total = (policy_row['attack_classified_as_utility'] + 
                              policy_row['attack_classified_as_ambiguous'] + 
                              policy_row['attack_classified_as_attack'] + 
                              policy_row['attack_classified_as_novel'])
            
            # Total ambiguous actions
            total_ambiguous = (policy_row['utility_classified_as_ambiguous'] + 
                              policy_row['attack_classified_as_ambiguous'])
            
            # Total misclassified actions
            total_misclassified = (policy_row['utility_classified_as_attack'] +  # Utility wrongly called attack
                                  policy_row['attack_classified_as_utility'])   # Attack wrongly called utility
            
            # Total actions
            total_actions = util_class_total + att_class_total
            
            # Weighted rates
            weighted_ambiguity = (total_ambiguous / total_actions) * 100 if total_actions > 0 else 0
            weighted_misclass = (total_misclassified / total_actions) * 100 if total_actions > 0 else 0
            
            print(f"{agent:<12} {frac:>3}%{' '*3} {weighted_ambiguity:>6.2f}%{' '*12} {weighted_misclass:>6.2f}%")
    
    print(f"{'='*90}\n")
    print("Note:")
    print("  - Correct = % of actions correctly classified (Utility→Utility, Attack→Attack)")
    print("  - Ambiguous = % of actions classified as ambiguous/borderline cases")
    print("  - Wrong = % of actions misclassified (Utility→Attack, Attack→Utility)")
    print("  - Each row in tables 1&2 should sum to 100%")
    print("  - Overall Ambiguity = (Total Ambiguous Actions) / (Total Actions) * 100")
    print("  - Overall Misclassification = (Total Misclassified Actions) / (Total Actions) * 100\n")

if __name__ == "__main__":
    main()
