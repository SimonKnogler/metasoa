#!/usr/bin/env python3
"""
Test script to visualize counterbalancing assignments
Run this to see what condition each participant ID gets
"""

import hashlib

def get_counterbalancing(participant_id):
    """Get counterbalancing assignment for a participant ID"""
    
    # Convert participant ID to integer
    try:
        participant_num = int(participant_id)
    except ValueError:
        # For non-numeric IDs, hash to get a consistent number
        participant_num = int(hashlib.sha256(participant_id.encode()).hexdigest(), 16) & 0xFFFF
    
    # Create counterbalancing index (0-7 for 8 conditions)
    cb_index = participant_num % 8
    
    # Factor 1: Learning order
    learning_order_first_angle = 0 if (cb_index & 1) == 0 else 90
    learning_order = [learning_order_first_angle, 90 if learning_order_first_angle == 0 else 0]
    
    # Factor 2: Which palette for first angle
    palette_first_is_blue_green = ((cb_index >> 1) & 1) == 0
    PALETTE_SET_1 = ("blue", "green")
    PALETTE_SET_2 = ("red", "yellow")
    
    if palette_first_is_blue_green:
        PALETTE_FOR_FIRST_ANGLE = PALETTE_SET_1
        PALETTE_FOR_SECOND_ANGLE = PALETTE_SET_2
    else:
        PALETTE_FOR_FIRST_ANGLE = PALETTE_SET_2
        PALETTE_FOR_SECOND_ANGLE = PALETTE_SET_1
    
    # Factor 3: Color-difficulty mapping within first palette
    first_palette_flip = ((cb_index >> 2) & 1) == 1
    if first_palette_flip:
        PALETTE_FOR_FIRST_ANGLE = (PALETTE_FOR_FIRST_ANGLE[1], PALETTE_FOR_FIRST_ANGLE[0])
    
    # Second angle always flips to ensure variety
    PALETTE_FOR_SECOND_ANGLE = (PALETTE_FOR_SECOND_ANGLE[1], PALETTE_FOR_SECOND_ANGLE[0])
    
    return {
        'cb_index': cb_index,
        'learning_order': learning_order,
        'first_angle_palette': PALETTE_FOR_FIRST_ANGLE,
        'second_angle_palette': PALETTE_FOR_SECOND_ANGLE
    }

# Test all 8 conditions using participant IDs 1-8
print("=" * 80)
print("COUNTERBALANCING TEST - Participant IDs 1-8")
print("=" * 80)

for p_id in range(1, 9):
    result = get_counterbalancing(str(p_id))
    print(f"\nParticipant {p_id}:")
    print(f"  CB Index: {result['cb_index']}")
    print(f"  Learning Order: {result['learning_order']}")
    print(f"  First Angle ({result['learning_order'][0]}°): {result['first_angle_palette'][0]} (hard) / {result['first_angle_palette'][1]} (easy)")
    print(f"  Second Angle ({result['learning_order'][1]}°): {result['second_angle_palette'][0]} (hard) / {result['second_angle_palette'][1]} (easy)")

# Test with text IDs
print("\n" + "=" * 80)
print("TEXT ID EXAMPLES (will always get same assignment)")
print("=" * 80)

for text_id in ["test", "SIM", "pilot", "demo"]:
    result = get_counterbalancing(text_id)
    print(f"\nParticipant '{text_id}':")
    print(f"  CB Index: {result['cb_index']}")
    print(f"  Learning Order: {result['learning_order']}")
    print(f"  First: {result['first_angle_palette']} @ {result['learning_order'][0]}°")
    print(f"  Second: {result['second_angle_palette']} @ {result['learning_order'][1]}°")

print("\n" + "=" * 80)
print("VERIFY: Run this script multiple times - same ID always gets same assignment!")
print("=" * 80)

