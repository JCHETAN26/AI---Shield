#!/usr/bin/env python3
"""
Direct mitigation test that bypasses Flask session issues.
This will test the mitigation system directly through the web interface.
"""

import requests
import json
import time

BASE_URL = "http://localhost:5001"

def test_complete_mitigation_workflow():
    """Test complete mitigation workflow from scratch."""
    print("🛡️ Testing Complete Mitigation Workflow")
    print("=" * 50)
    
    try:
        # Step 1: Start a demo analysis first
        print("📊 Step 1: Starting demo analysis...")
        demo_response = requests.get(f"{BASE_URL}/configure/demo")
        if demo_response.status_code != 200:
            print(f"❌ Demo start failed: {demo_response.status_code}")
            return False
        
        # Extract session ID from response
        demo_text = demo_response.text
        import re
        session_match = re.search(r'demo_[a-f0-9]+', demo_text)
        if not session_match:
            print("❌ Could not extract demo session ID")
            return False
        
        demo_session_id = session_match.group(0)
        print(f"✅ Demo session created: {demo_session_id}")
        
        # Step 2: Start the analysis
        print("🚀 Step 2: Starting analysis...")
        analysis_data = {
            'session_id': demo_session_id,
            'fgsm_epsilon': '0.1',
            'pgd_epsilon': '0.1',
            'pgd_alpha': '0.01',
            'pgd_iterations': '40',
            'max_samples': '100',
            'include_shap': 'on',
            'include_lime': 'on'
        }
        
        start_response = requests.post(f"{BASE_URL}/start_analysis", data=analysis_data)
        if start_response.status_code != 200:
            print(f"❌ Analysis start failed: {start_response.status_code}")
            print(f"Response: {start_response.text[:500]}")
            return False
        
        print("✅ Analysis started!")
        
        # Step 3: Wait for demo analysis to complete
        print("⏱️ Step 2: Waiting for analysis to complete...")
        max_wait = 180  # 3 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{BASE_URL}/status/{demo_session_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"   Progress: {status_data.get('progress', 0)}% - {status_data.get('message', 'Processing...')}")
                
                if status_data.get('status') == 'completed':
                    print("✅ Analysis completed!")
                    break
                elif status_data.get('status') == 'failed':
                    print(f"❌ Analysis failed: {status_data.get('error', 'Unknown error')}")
                    return False
            
            time.sleep(3)
        else:
            print("❌ Analysis timed out")
            return False
        
        # Step 4: Start mitigation
        print("🛡️ Step 4: Starting mitigation...")
        mitigation_data = {
            'strategies': ['adversarial_training', 'feature_preprocessing', 'ensemble_defense']
        }
        
        mitigation_response = requests.post(
            f"{BASE_URL}/mitigate/{demo_session_id}",
            data=mitigation_data
        )
        
        if mitigation_response.status_code != 200:
            print(f"❌ Mitigation start failed: {mitigation_response.status_code}")
            print(f"Response: {mitigation_response.text[:500]}")
            return False
        
        # Extract mitigation session ID
        mitigation_text = mitigation_response.text
        mitigation_session_match = re.search(r'demo_[a-f0-9]+_mitigation', mitigation_text)
        if not mitigation_session_match:
            print("❌ Could not extract mitigation session ID")
            return False
        
        mitigation_session_id = mitigation_session_match.group(0)
        print(f"✅ Mitigation session created: {mitigation_session_id}")
        
        # Step 5: Monitor mitigation progress
        print("⏱️ Step 5: Monitoring mitigation progress...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{BASE_URL}/status/{mitigation_session_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"   Progress: {status_data.get('progress', 0)}% - {status_data.get('message', 'Processing...')}")
                
                if status_data.get('status') == 'completed':
                    print("✅ Mitigation completed!")
                    
                    # Show results
                    results = status_data.get('results', {})
                    if results:
                        print("\n📋 Mitigation Results:")
                        mitigation_results = results.get('mitigation_results', {})
                        for strategy, result in mitigation_results.items():
                            success = '✅' if result.get('success', False) else '❌'
                            improvement = result.get('robustness_improvement', 0)
                            accuracy = result.get('hardened_accuracy', 0)
                            print(f"   {success} {strategy}: {improvement:.2%} improvement, {accuracy:.2%} accuracy")
                        
                        summary = results.get('summary', {})
                        print(f"\n📊 Summary:")
                        print(f"   • Total strategies: {summary.get('total_strategies', 0)}")
                        print(f"   • Successful: {summary.get('successful_strategies', 0)}")
                        print(f"   • Best strategy: {summary.get('best_strategy', 'N/A')}")
                        print(f"   • Best improvement: {summary.get('best_improvement', 0):.2%}")
                        
                        recommendations = summary.get('recommendations', [])
                        if recommendations:
                            print(f"\n💡 Recommendations:")
                            for rec in recommendations[:3]:  # Show top 3
                                print(f"   • {rec}")
                    
                    return True
                    
                elif status_data.get('status') == 'failed':
                    print(f"❌ Mitigation failed: {status_data.get('error', 'Unknown error')}")
                    print(f"Message: {status_data.get('message', 'No message')}")
                    return False
            else:
                print(f"⚠️ Status check returned {status_response.status_code}")
                if status_response.status_code == 404:
                    print("   Session may have been lost due to server restart")
                    return False
            
            time.sleep(2)
        
        print("❌ Mitigation timed out")
        return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_mitigation_workflow()
    if success:
        print("\n🎉 Complete mitigation workflow test PASSED!")
    else:
        print("\n💥 Complete mitigation workflow test FAILED!")
        exit(1)