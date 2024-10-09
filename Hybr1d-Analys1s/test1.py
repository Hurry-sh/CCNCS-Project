from mosf_client import MobSF
import json
import os

# Define the APK file path
apk_path = "/home/mrgrey/Downloads/anilab-latest.apk"
output_dir = "mobsf_analysis_results_Anilab"

def ensure_output_directory():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def save_to_file(filename, content):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(content, f, indent=4)
    print(f"Saved {filename}")

def print_security_summary(report):
    print("\n=== SECURITY ANALYSIS SUMMARY ===\n")
    
    # App Information
    print(f"App Name: {report.get('app_name', 'Not found')}")
    print(f"Package Name: {report.get('package_name', 'Not found')}")
    print(f"Version: {report.get('version', 'Not found')}")
    
    # Vulnerability Information
    print("\n--- Vulnerability Information ---")
    cvss_score = report.get('average_cvss')
    print(f"Average CVSS Score: {cvss_score if cvss_score is not None else 'Not available'}")
    
    cve_list = report.get('CVE', [])
    if cve_list and isinstance(cve_list, list):
        print("\nCVE Vulnerabilities Found:")
        for cve in cve_list:
            print(f"- {cve}")
    else:
        print("No CVE vulnerabilities found.")
    
    # Code Analysis Summary
    print("\n--- Code Analysis Summary ---")
    code_analysis = report.get('code_analysis')
    if isinstance(code_analysis, dict):
        high_risks = [k for k, v in code_analysis.items() 
                     if isinstance(v, dict) and v.get('severity') == 'high']
        if high_risks:
            print(f"High Risk Findings: {len(high_risks)} issues found")
        else:
            print("No high-risk findings in code analysis.")
    else:
        print(f"Code analysis status: {code_analysis}")
    
    # Permissions Summary
    print("\n--- Permissions Summary ---")
    malware_permissions = report.get('malware_permissions', [])
    if malware_permissions and isinstance(malware_permissions, list):
        print(f"Suspicious Permissions: {len(malware_permissions)} found")
    else:
        print("No suspicious permissions found.")
    
    # Security Score
    security_score = report.get('security_score', 'Not available')
    print(f"\nOverall Security Score: {security_score}")

def main():
    try:
        ensure_output_directory()
        
        # Create an instance of MobSF
        mobsf = MobSF(
            apikey="09fc18eedc7d8269bb586ea346b5e397e8690c342e5dc1fe03bb31eb127ba209",
            server="http://localhost:8000/"
        )
        
        # Upload and scan file
        print("Starting analysis...")
        with open(apk_path, "rb") as file:
            response = mobsf.upload("Spotify.apk", file)
            if not response:
                print("Upload failed")
                return
        
        scan_response = mobsf.scan(mobsf.hash)
        report = mobsf.report_json(mobsf.hash)
        
        # Save all detailed information to files
        
        # 1. General Information
        general_info = {key: report.get(key, "Not found") for key in 
                        ['title', 'version', 'file_name', 'app_name', 'app_type', 'size', 'package_name']}
        save_to_file("1_general_info.json", general_info)
        
        # 2. Hashes
        hashes = {hash_type: report.get(hash_type, "Not found") for hash_type in 
                 ['md5', 'sha1', 'sha256']}
        save_to_file("2_hashes.json", hashes)
        
        # 3. Vulnerability Information
        vulnerability_info = {
            "average_cvss": report.get('average_cvss', 'Not available'),
            "CVE": report.get('CVE', 'Not available')
        }
        save_to_file("3_vulnerability_info.json", vulnerability_info)
        
        # 4. SDK Information
        sdk_info = {
            "target_sdk": report.get('target_sdk', 'Not found'),
            "min_sdk": report.get('min_sdk', 'Not found'),
            "max_sdk": report.get('max_sdk', 'Not found')
        }
        save_to_file("4_sdk_info.json", sdk_info)
        
        # 5. Security Information
        security_info = {
            "permissions": report.get('permissions', []),
            "malware_permissions": report.get('malware_permissions', []),
            "certificate_analysis": report.get('certificate_analysis', {}),
            "security_score": report.get('security_score', 'Not available'),
            "trackers": report.get('trackers', {}),
            "domain_malware": report.get('domain_malware', {}),
            "exported_count": report.get('exported_count', {}),
            "binary_analysis": report.get('binary_analysis', {})
        }
        save_to_file("5_security_info.json", security_info)
        
        # 6. Content Analysis
        content_analysis = {
            "urls": report.get('urls', []),
            "emails": report.get('emails', []),
            "firebase_urls": report.get('firebase_urls', []),
            "files": report.get('files', {}),
            "strings": report.get('strings', [])
        }
        save_to_file("6_content_analysis.json", content_analysis)
        
        # 7. Code Analysis
        code_analysis = {
            "code_analysis": report.get('code_analysis', {})
        }
        save_to_file("7_code_analysis.json", code_analysis)
        
        # 8. Full Report (for reference)
        save_to_file("8_full_report.json", report)
        
        # Print security summary to terminal
        print_security_summary(report)
        
        print(f"\nDetailed analysis results saved in '{output_dir}' directory")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()