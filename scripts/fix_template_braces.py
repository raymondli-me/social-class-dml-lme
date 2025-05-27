#!/usr/bin/env python3
"""
Fix the template string in analyze_dml_top_pcs_umap_updated.py by escaping curly braces
"""

import re

# Read the file
with open('analyze_dml_top_pcs_umap_updated.py', 'r') as f:
    content = f.read()

# Find the html_template string
template_start = content.find('html_template = """')
template_end = content.find('"""', template_start + 20)

if template_start == -1 or template_end == -1:
    print("Could not find html_template")
    exit(1)

# Extract the template
template = content[template_start:template_end + 3]

# Find all the format placeholders we want to keep
placeholders = [
    '{comparison_table_rows}',
    '{top_pcs_list}',
    '{variance_explained}',
    '{n_points}',
    '{pc_controls}',
    '{pc_color_options}',
    '{data_json}',
    '{top_pcs_json}',
    '{pc_importance_json}'
]

# Replace placeholders with temporary markers
temp_template = template
for i, placeholder in enumerate(placeholders):
    temp_template = temp_template.replace(placeholder, f'__PLACEHOLDER_{i}__')

# Now escape all remaining curly braces
temp_template = temp_template.replace('{', '{{').replace('}', '}}')

# Restore the placeholders
for i, placeholder in enumerate(placeholders):
    temp_template = temp_template.replace(f'__PLACEHOLDER_{i}__', placeholder)

# Replace in the original content
new_content = content[:template_start] + temp_template + content[template_end + 3:]

# Write the fixed file
with open('analyze_dml_top_pcs_umap_fixed.py', 'w') as f:
    f.write(new_content)

print("Fixed file created: analyze_dml_top_pcs_umap_fixed.py")
print("The template string now has properly escaped curly braces.")