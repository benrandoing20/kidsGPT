import os
import re

out = open("scrape/storybooks_kids.txt", "w", encoding="utf-8")

for file in os.listdir("scrape/sbc-source/en/"):
    if file.endswith(".md"):
        with open(f"scrape/sbc-source/en/{file}", "r", encoding="utf-8") as f:
            content = f.read()
            
            # Split into lines and process
            lines = content.split('\n')
            story_lines = []
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip markdown headers (# and ##)
                if line.startswith('#') or line.startswith('##'):
                    continue
                
                # Skip metadata lines (License, Text, Illustration, Language)
                if line.startswith('* License:') or line.startswith('* Text:') or line.startswith('* Illustration:') or line.startswith('* Language:'):
                    continue
                
                # Skip lines that are just asterisks or other formatting
                if line == '*' or line == '**' or re.match(r'^\*+$', line):
                    continue
                
                # Only include lines with actual story content (more than 3 characters)
                if len(line) > 3:
                    story_lines.append(line)
            
            # Write the story content
            if story_lines:
                out.write('\n'.join(story_lines) + '\n\n')

out.close()
print("Processing complete!")
