#!/usr/bin/env python3
"""
Test End-to-End Mitigation Functionality
Tests the complete mitigation workflow through the web interface.
"""

import requests
import json
import time
import os
import sys

def test_mitigation_workflow():
    """Test the complete mitigation workflow."""
    print("🛡️ Testing AI Shield Mitigation Workflow")
    print("=" * 50)
    
    base_url = "http://localhost:5001"
    
    # Step 1: Check if Flask app is running
    try:
        response = requests.get(base_url)
        if response.status_code != 200:
            print("❌ Flask app not running on port 5001")
            return False
        print("✅ Flask app is running")
    except Exception as e:
        print(f"❌ Failed to connect to Flask app: {e}")
        return False
    
    # Step 2: List available sessions (should have demo data)
    try:
        sessions_response = requests.get(f"{base_url}/api/sessions")
        sessions = sessions_response.json()
        print(f"📊 Available sessions: {len(sessions)}")
        
        # Look for a completed session to use for mitigation
        completed_sessions = [s for s in sessions if s.get('status') == 'completed']
        if not completed_sessions:
            print("❌ No completed sessions found. Run analysis first.")
            return False
        
        session_id = completed_sessions[0]['session_id']
        print(f"✅ Using session: {session_id}")
        
    except Exception as e:
        print(f"❌ Failed to get sessions: {e}")
        return False
    
    # Step 3: Start mitigation workflow
    try:
        mitigation_data = {
            'strategies': ['adversarial_training', 'ensemble_defense', 'feature_preprocessing']
        }
        
        response = requests.post(f"{base_url}/mitigate/{session_id}", 
                               json=mitigation_data)
        
        if response.status_code != 200:
            print(f"❌ Failed to start mitigation: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        mitigation_response = response.json()
        mitigation_session_id = mitigation_response.get('mitigation_session_id')
        
        if not mitigation_session_id:
            print("❌ No mitigation session ID returned")
            return False
        
        print(f"✅ Started mitigation session: {mitigation_session_id}")
        
    except Exception as e:
        print(f"❌ Failed to start mitigation: {e}")
        return False
    
    # Step 4: Monitor progress
    print("⏳ Monitoring mitigation progress...")
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            status_response = requests.get(f"{base_url}/status/{mitigation_session_id}")
            status_data = status_response.json()
            
            progress = status_data.get('progress', 0)
            status = status_data.get('status', 'unknown')
            message = status_data.get('message', 'No message')
            
            print(f"📈 Progress: {progress}% - {message}")
            
            if status == 'completed':
                print("✅ Mitigation completed successfully!")
                break
            elif status == 'failed':
                error = status_data.get('error', 'Unknown error')
                print(f"❌ Mitigation failed: {error}")
                return False
            
            time.sleep(2)
            attempt += 1
            
        except Exception as e:
            print(f"⚠️ Error checking status: {e}")
            attempt += 1
    
    if attempt >= max_attempts:
        print("⏰ Timeout waiting for mitigation to complete")
        return False
    
    # Step 5: Retrieve results
    try:
        results_response = requests.get(f"{base_url}/results/{mitigation_session_id}")
        results = results_response.json()
        
        print("📊 Mitigation Results:")
        print("-" * 30)
        
        if 'results' in results and 'mitigation_results' in results['results']:
            mitigation_results = results['results']['mitigation_results']
            summary = results['results'].get('summary', {})
            
            print(f"Total strategies tested: {summary.get('total_strategies', 0)}")
            print(f"Successful strategies: {summary.get('successful_strategies', 0)}")
            print(f"Best strategy: {summary.get('best_strategy', 'None')}")
            print(f"Best improvement: {summary.get('best_improvement', 0):.2%}")
            
            print("\nStrategy Details:")
            for strategy, result in mitigation_results.items():
                success = "✅" if result.get('success', False) else "❌"
                improvement = result.get('robustness_improvement', 0)
                print(f"{success} {strategy}: {improvement:.2%} improvement")
            
            # Check for recommendations
            recommendations = summary.get('recommendations', [])
            if recommendations:
                print("\n💡 Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec}")
        
        print("\n✅ Mitigation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to retrieve results: {e}")
        return False

def main():
    """Run the mitigation test."""
    success = test_mitigation_workflow()
    
    if success:
        print("\n🎉 All mitigation tests passed!")
        print("🛡️ AI Shield mitigation system is working correctly!")
    else:
        print("\n💥 Mitigation tests failed!")
        print("🔧 Please check the system and try again.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)