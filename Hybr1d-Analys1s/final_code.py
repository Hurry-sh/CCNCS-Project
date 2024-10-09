from mosf_client import MobSF
import json
import os
import subprocess

# Define the APK file path
apk_path = "/home/mrgrey/Downloads/Spotify.apk"

def format_security_analysis(report):
    output = []

    # PERMISSIONS AND SECURITY
    output.append("# PERMISSIONS AND SECURITY\n")

    # Permissions
    output.append("## Permissions")
    for perm, details in report['permissions'].items():
        output.append(f"- {perm}")
        output.append(f"  - Status: {details['status']}")
        output.append(f"  - Info: {details['info']}")
        output.append(f"  - Description: {details['description']}\n")

    # Malware Permissions
    output.append("## Malware Permissions")
    mp = report['malware_permissions']
    output.append(f"- Top malware permissions: {mp['top_malware_permissions']}")
    output.append(f"- Other abused permissions: {mp['other_abused_permissions']}")
    output.append(f"- Total malware permissions: {mp['total_malware_permissions']}")
    output.append(f"- Total other permissions: {mp['total_other_permissions']}\n")

    # Certificate Analysis
    output.append("## Certificate Analysis")
    ca = report['certificate_analysis']
    output.append(f"- Certificate info: {ca['certificate_info']}\n")
    output.append("### Certificate Findings")
    for finding in ca['certificate_findings']:
        output.append(f"- [{finding[0].upper()}] {finding[1]}")
    output.append("\n### Certificate Summary")
    for key, value in ca['certificate_summary'].items():
        output.append(f"- {key.capitalize()}: {value}")

    output.append("")

    # Permission Mapping
    output.append("## Permission Mapping")
    for perm, locations in report['permission_mapping'].items():
        output.append(f"- {perm}:")
        for file, lines in locations.items():
            output.append(f"  - {file}: {lines}")
    output.append("")

    # Exported Components Count
    output.append("## Exported Components Count")
    for key, value in report['exported_count'].items():
        output.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
    output.append("")

    # CONTENT ANALYSIS
    output.append("# CONTENT ANALYSIS\n")
    for section in ['urls', 'domains', 'emails']:
        output.append(f"## {section.capitalize()}")
        if report[section]:
            for item in report[section]:
                output.append(f"- {item}")
        else:
            output.append("None found")
        output.append("")

    # Files
    output.append("## Files")
    for i, file in enumerate(report['files'], 1):
        output.append(f"{i}. {file}")
    output.append("")

    # CODE ANALYSIS
    output.append("# CODE ANALYSIS\n")
    output.append("## Findings")
    for finding, details in report['Code_analysis']['findings'].items():
        output.append(f"1. {finding.replace('_', ' ').capitalize()}")
        output.append(f" - Files: {', '.join([f'{file} (line {line})' for file, line in details['files'].items()])}")
        output.append(f" - Severity: {details['metadata']['severity'].capitalize()}")
        output.append(f" - CVSS: {details['metadata']['cvss']}")
        output.append(f" - CWE: {details['metadata']['cwe']}")
        output.append(f" - MASVS: {details['metadata']['masvs']}")
        output.append(f" - Description: {details['metadata']['description']}\n")

    output.append("## Summary")
    for key, value in report['Code_analysis']['summary'].items():
        output.append(f"- {key.capitalize()}: {value}")

    return "\n".join(output)
def format_security_analysis(report):
    output = []

    # PERMISSIONS AND SECURITY
    output.append("# PERMISSIONS AND SECURITY\n")

    # Permissions
    output.append("## Permissions")
    for perm, details in report.get('permissions', {}).items():
        output.append(f"- {perm}")
        output.append(f"  - Status: {details['status']}")
        output.append(f"  - Info: {details['info']}")
        output.append(f"  - Description: {details['description']}\n")

    # Malware Permissions
    output.append("## Malware Permissions")
    mp = report.get('malware_permissions', {})
    output.append(f"- Top malware permissions: {mp.get('top_malware_permissions', 'None')}")
    output.append(f"- Other abused permissions: {mp.get('other_abused_permissions', 'None')}")
    output.append(f"- Total malware permissions: {mp.get('total_malware_permissions', 0)}")
    output.append(f"- Total other permissions: {mp.get('total_other_permissions', 0)}\n")

    # Certificate Analysis
    output.append("## Certificate Analysis")
    ca = report.get('certificate_analysis', {})
    output.append(f"- Certificate info: {ca.get('certificate_info', 'Not available')}\n")
    output.append("### Certificate Findings")
    for finding in ca.get('certificate_findings', []):
        output.append(f"- [{finding[0].upper()}] {finding[1]}")
    output.append("\n### Certificate Summary")
    for key, value in ca.get('certificate_summary', {}).items():
        output.append(f"- {key.capitalize()}: {value}")

    output.append("")

    # Permission Mapping
    output.append("## Permission Mapping")
    for perm, locations in report.get('permission_mapping', {}).items():
        output.append(f"- {perm}:")
        for file, lines in locations.items():
            output.append(f"  - {file}: {lines}")
    output.append("")

    # Exported Components Count
    output.append("## Exported Components Count")
    for key, value in report.get('exported_count', {}).items():
        output.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
    output.append("")

    # Files
    output.append("## Files")
    for i, file in enumerate(report.get('files', []), 1):
        output.append(f"{i}. {file}")
    output.append("")

    # CODE ANALYSIS
    output.append("# CODE ANALYSIS\n")

    output.append("## Findings")
    code_analysis = report.get('Code_analysis', {})
    for finding, details in code_analysis.get('findings', {}).items():
        output.append(f"1. {finding.replace('_', ' ').capitalize()}")
        output.append(f" - Files: {', '.join([f'{file} (line {line})' for file, line in details['files'].items()])}")
        output.append(f" - Severity: {details['metadata']['severity'].capitalize()}")
        output.append(f" - CVSS: {details['metadata']['cvss']}")
        output.append(f" - CWE: {details['metadata']['cwe']}")
        output.append(f" - MASVS: {details['metadata']['masvs']}")
        output.append(f" - Description: {details['metadata']['description']}\n")
    # CODE ANALYSIS
    output.append("# CODE ANALYSIS\n")

    output.append("## Findings")
    code_analysis = report.get('Code_analysis', {})
    for finding, details in code_analysis.get('findings', {}).items():
        output.append(f"### {finding.replace('_', ' ').capitalize()}")
        output.append(f" - Files: {', '.join([f'{file} (line {line})' for file, line in details['files'].items()])}")
        output.append(f" - Severity: **{details['metadata']['severity'].capitalize()}**")
        output.append(f" - CVSS: {details['metadata']['cvss']}")
        output.append(f" - CWE: {details['metadata']['cwe']}")
        output.append(f" - MASVS: {details['metadata']['masvs']}")
        output.append(f" - Description: {details['metadata']['description']}\n")

    output.append("## Summary")
    for key, value in code_analysis.get('summary', {}).items():
        output.append(f"- {key.capitalize()}: {value}")

    return "\n".join(output)
    

    output.append("## Summary")
    for key, value in code_analysis.get('summary', {}).items():
        output.append(f"- {key.capitalize()}: {value}")

    return "\n".join(output)

def main():
    # Create an instance of MobSF
    mobsf = MobSF(apikey="b399480d452c72911a55d2d152253b6a30382887077f630642f82a02c8075ada", server="http://localhost:8000/")

    # Upload a file (ensure you have a valid file path)
    with open("/home/mrgrey/Downloads/Spotify.apk", "rb") as file:
        response = mobsf.upload("Spotify.apk", file)

    print(response)

    # If upload was successful, proceed with scanning
    if response:
        scan_response = mobsf.scan(mobsf.hash)

        # Get JSON report of the scan
        report = mobsf.report_json(mobsf.hash)

        # Print general app information
        print("# GENERAL APP INFORMATION")
        gen = ['title', 'version', 'file_name', 'app_name', 'app_type', 'size', 'package_name']

        for key in gen:
            if key in report:
                print(f"{key}: {report[key]}")
            else:
                print(f"{key}: Key '{key}' not found in report")

        print("---------------------------------------------------------------")

        print("## HASHES")
        hashes = ['md5', 'sha1', 'sha256']

        for hash in hashes:
            if hash in report:
                print(f"{hash}: {report[hash]}")
            else:
                print(f"{hash}: Key '{hash}' not found in report")

        print("---------------------------------------------------------------")

        print("Average CVSS:", report.get('average_cvss', 'No {Common Vulnerability Scoring System} info available.'))
        print("CVE:", report.get('CVE', 'No CVE information available.'))

        print("### SDK INFORMATION:")
        sdks = ['target_sdk', 'max_sdk', 'min_sdk']

        for sdk in sdks:
            if sdk in report:
                print(f"{sdk}: {report[sdk]}")
            else:
                print(f"{sdk}: Key '{sdk}' not found in report")

        print("-------------------------------------------")

        print("#### PERMISSIONS AND SECURITY:")
        security = ['permissions', 'malware_permissions', 'certificate_analysis', 'permission_mapping', 'exported_count']

        for per in security:
            if per in report:
                print(f'{per}: {report[per]}')
            else:
                print(f"{per}: Key '{per}' not found in report")

        print("----------------------------------------------------")

        print("##### CONTENT ANALYSIS:")
        content = ['urls', 'domains', 'emails', 'strings', 'firebase_urls', 'files']

        for url in content:
            print(f'{url}: {report[url]}')

        print("-----------------------------------------")

        print("##### Code analysis:", report.get('code_analysis', "Too much obfuscation (in simple meaning -> packing of strings)"))
        print("-----------------------------------------")

        # Format and print the security analysis
        formatted_output = format_security_analysis(report)
        print(formatted_output)
      
if __name__ == "__main__":
    main()