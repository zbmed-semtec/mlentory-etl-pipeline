import os
import torch
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import hashlib

@dataclass
class Section:
    """Represents a section of text with its title and content"""

    title: str
    content: str
    start_idx: int
    end_idx: int

    def to_dict(self) -> Dict[str, any]:
        """Convert Section to dictionary for serialization"""
        return {
            "title": self.title,
            "content": self.content, 
            "start_idx": self.start_idx,
            "end_idx": self.end_idx
        }

    def to_json(self) -> str:
        """Convert Section to JSON string"""
        import json
        return json.dumps(self.to_dict(), indent=4)


class MarkdownParser:
    """
    A parser for markdown files that extracts structured sections and handles special markdown elements.
    Provides methods to extract header-based sections and fine-grained sections from markdown text.
    """
    
    def __init__(self):
        """Initialize the markdown parser."""
        pass
    
    def extract_sections(self, text: str) -> List[Section]:
        """
        Extract sections from text based on markdown-style headers, maintaining header hierarchy.
        Intelligently distinguishes between actual markdown headers and code comments or other
        occurrences of # symbols.

        Args:
            text (str): The text to segment

        Returns:
            List[Section]: List of extracted sections with hierarchical titles and content

        Example:
            >>> parser = MarkdownParser()
            >>> sections = parser.extract_sections("# Header\\nContent\\n```python\\n# Not a header\\n```")
            >>> len(sections)
            1
        """
        # Split text into lines
        lines = text.split("\n")
        sections = []
        current_title = ""
        current_content = []
        start_idx = 0
        # Keep track of header hierarchy - using list of (level, title) tuples
        # Lower levels (h1) have smaller numbers, higher levels (h6) have larger numbers
        header_hierarchy = []
        
        # Track if we're inside a code block
        in_code_block = False
        code_block_markers = ["```", "~~~"]  # Common markdown code block delimiters
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Check if entering or exiting a code block
            if any(stripped_line.startswith(marker) for marker in code_block_markers):
                in_code_block = not in_code_block
                current_content.append(line)
                continue
                
            # Skip header detection if inside a code block
            if in_code_block:
                current_content.append(line)
                continue
                
            # More robust markdown header detection:
            # 1. Must start at beginning of line (no indentation)
            # 2. Must have space after hash symbols
            # 3. Cannot be inside a code block
            header_match = None
            if not stripped_line.startswith("#"):
                # Fast path - not a header candidate
                current_content.append(line)
                continue
                
            # Only check if the line actually starts with # at the beginning
            if line.lstrip() == stripped_line:  # No indentation
                header_match = re.match(r"^(#{1,6})\s+(.+)$", stripped_line)
            
            # Process the line
            is_last_line = (i == len(lines) - 1)
            
            if header_match or is_last_line:
                # Save previous section if exists and has content
                if current_title and current_content:
                    # Remove empty lines and check if there's actual content
                    filtered_content = [l for l in current_content if l.strip()]
                    if filtered_content:
                        sections.append(
                            Section(
                                title=current_title,
                                content=current_title
                                + ":\n"
                                + "\n".join(filtered_content),
                                start_idx=start_idx,
                                end_idx=i,
                            )
                        )

                if header_match:
                    level = len(header_match.group(1))  # Number of # symbols
                    title = header_match.group(2).strip()

                    # Update header hierarchy based on current level
                    # Remove any headers of equal or higher level (smaller or equal numbers)
                    header_hierarchy = [h for h in header_hierarchy if h[0] < level]
                    # Add the current header to the hierarchy
                    header_hierarchy.append((level, title))
                    
                    # Build the hierarchical title from the current hierarchy
                    # Sort by level to get the correct order (h1 -> h2 -> h3)
                    sorted_hierarchy = sorted(header_hierarchy)
                    current_title = " > ".join(title for _, title in sorted_hierarchy)
                    
                    current_content = []
                    start_idx = i
                elif is_last_line and not header_match:
                    current_content.append(line)
            else:
                current_content.append(line)

        # Process any remaining content after the last iteration
        # This handles the case where there's content after the last header without another header following
        if current_title and current_content:
            # Remove empty lines and check if there's actual content
            filtered_content = [l for l in current_content if l.strip()]
            if filtered_content:
                sections.append(
                    Section(
                        title=current_title,
                        content=current_title
                        + ":\n"
                        + "\n".join(filtered_content),
                        start_idx=start_idx,
                        end_idx=len(lines) - 1,  # End at the last line
                    )
                )
        elif current_content:
            # Remove empty lines and check if there's actual content
            filtered_content = [l for l in current_content if l.strip()]
            if filtered_content:
                sections.append(
                    Section(
                        title="",
                        content="\n".join(filtered_content),
                        start_idx=start_idx,
                        end_idx=len(lines) - 1,  # End at the last line
                    )
                )

        return sections
    
    def _is_special_markdown_block(self, lines: List[str], start_idx: int) -> Tuple[bool, int]:
        """
        Detect if lines starting at start_idx form a special markdown block that should be kept intact.
        
        Args:
            lines (List[str]): List of text lines
            start_idx (int): Index to start checking from
            
        Returns:
            Tuple[bool, int]: (is_special_block, end_idx)
        """
        if start_idx >= len(lines):
            return False, start_idx
            
        current_line = lines[start_idx].strip()
        
        # Check for code blocks
        if current_line.startswith("```") or current_line.startswith("~~~"):
            marker = current_line[:3]
            # Find the end of the code block
            for i in range(start_idx + 1, len(lines)):
                if lines[i].strip().startswith(marker):
                    return True, i
            # If no end marker is found, treat the rest as a code block
            return True, len(lines) - 1
            
        # Check for tables (line containing | character)
        if "|" in current_line and not current_line.startswith(">"):
            # Find extent of table by looking for lines with | character
            table_end = start_idx
            for i in range(start_idx + 1, len(lines)):
                if "|" in lines[i].strip():
                    table_end = i
                elif lines[i].strip() == "":
                    break  # Empty line marks end of table
                else:
                    break  # Non-table line
            if table_end > start_idx:  # At least 2 lines with | character
                return True, table_end
                
        # Check for blockquotes (lines starting with >)
        if current_line.startswith(">"):
            blockquote_end = start_idx
            for i in range(start_idx + 1, len(lines)):
                if lines[i].strip().startswith(">") or lines[i].strip() == "":
                    blockquote_end = i
                else:
                    break
            if blockquote_end > start_idx:
                return True, blockquote_end
                
        # Check for lists (lines starting with -, *, +, or numbered)
        list_markers = ["-", "*", "+"]
        # Improved regex for ordered lists that matches digits followed by period and space
        numbered_pattern = re.compile(r"^\d+\.\s")
        
        # Check if current line is a list item (either unordered or ordered)
        is_list_item = (
            any(current_line.lstrip().startswith(marker + " ") for marker in list_markers) or
            numbered_pattern.match(current_line.lstrip()) is not None
        )
        
        if is_list_item:
            list_end = start_idx
            indentation_level = len(current_line) - len(current_line.lstrip())
            in_list = True
            i = start_idx + 1
            
            while i < len(lines) and in_list:
                next_line = lines[i].strip()
                next_line_indentation = len(lines[i]) - len(lines[i].lstrip()) if lines[i].strip() else 0
                
                # Check if this line is a list item or part of a list
                if next_line == "":
                    # Empty lines can be part of lists, but only if there is a list element after it
                    if i + 1 < len(lines) and any(lines[i + 1].lstrip().startswith(marker + " ") for marker in list_markers) or numbered_pattern.match(lines[i + 1].lstrip()) is not None:
                        list_end = i
                    else:
                        break
                elif any(next_line.lstrip().startswith(marker + " ") for marker in list_markers) or numbered_pattern.match(next_line.lstrip()) is not None:
                    # This is a list item - update the end marker
                    list_end = i
                    # Check if this is a sub-list item (more indented)
                    if next_line_indentation > indentation_level:
                        # Sublist item, continue
                        pass
                    elif next_line_indentation < indentation_level and next_line_indentation == 0:
                        # Back to main text with no indentation - end of list
                        break
                elif next_line_indentation > indentation_level:
                    # Indented continuation of previous list item
                    list_end = i
                elif next_line.startswith("#") or next_line.startswith("```") or next_line.startswith("~~~") or next_line.startswith(">"):
                    # Start of a header, code block, or blockquote - end of list
                    break
                else:
                    # Not clearly a list item, but might still be part of the list content
                    # if it's at the same indentation level or more indented
                    if next_line_indentation >= indentation_level:
                        list_end = i
                    else:
                        # Less indented non-list item - end of list
                        break
                
                i += 1
                    
            if list_end > start_idx:
                return True, list_end
                
        return False, start_idx
        
    def extract_fine_grained_sections(
        self, text: str, title: str = "", start_idx: int = 0, 
        end_idx: int = 0, max_section_length: int = 1000
    ) -> List[Section]:
        """
        Create fine-grained sections by splitting text into smaller chunks based on paragraphs and sentences.
        Used to further segment content within each header section.
        Keeps special markdown elements like tables, code blocks, blockquotes, and lists intact.

        Args:
            text (str): The text to segment
            title (str): Title prefix for the sections (typically the header title)
            start_idx (int): Starting line index in the original document
            end_idx (int): Ending line index in the original document
            max_section_length (int): Maximum length (in characters) for sections

        Returns:
            List[Section]: List of fine-grained sections

        Example:
            >>> parser = MarkdownParser()
            >>> sections = parser.extract_fine_grained_sections("Paragraph 1 with facts.\\n\\nParagraph 2 with different facts.", "Sample Title")
            >>> len(sections) >= 1
            True
        """
        lines = text.split("\n")
        sections = []
        
        # Find special blocks (tables, code blocks, blockquotes, lists) that should be kept intact
        special_blocks = []
        i = 0
        while i < len(lines):
            is_special, block_end = self._is_special_markdown_block(lines, i)
            if is_special:
                special_blocks.append((i, block_end))
                i = block_end + 1
            else:
                i += 1
                
        # Create regular paragraphs, making sure not to break special blocks
        paragraphs = []
        current_paragraph = []
        in_special_block = False
        special_block_start = -1
        special_block_end = -1
        
        for i, line in enumerate(lines):
            # Check if we're entering a special block
            for block_start, block_end in special_blocks:
                if i == block_start:
                    # End the current paragraph if exists
                    if current_paragraph and not in_special_block:
                        paragraphs.append(current_paragraph)
                        current_paragraph = []
                    
                    # Start a new paragraph for the special block
                    in_special_block = True
                    special_block_start = block_start
                    special_block_end = block_end
                    current_paragraph = [(j, lines[j]) for j in range(block_start, block_end + 1)]
                    
                    # Add it immediately as a paragraph
                    paragraphs.append(current_paragraph)
                    current_paragraph = []
                    break
                    
            # Skip if we're in a special block
            if in_special_block:
                if i > special_block_end:
                    in_special_block = False
                    special_block_start = -1
                    special_block_end = -1
                continue
                
            # Regular paragraph processing
            if line.strip():
                current_paragraph.append((i, line))
            elif current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = []
                
        # Add the last paragraph if not empty
        if current_paragraph:
            paragraphs.append(current_paragraph)
            
        # Create sections from paragraphs
        for p_idx, paragraph in enumerate(paragraphs):
            if not paragraph:
                continue
                
            paragraph_content = "\n".join(line for _, line in paragraph)
            
            # Calculate relative line indices within this text chunk
            relative_start = paragraph[0][0]
            relative_end = paragraph[-1][0]
            
            # Convert to absolute indices in the original document
            absolute_start = start_idx + relative_start
            absolute_end = start_idx + relative_end
            
            # Skip tiny paragraphs (unless they're a special block, indicated by significantly larger line count compared to character count)
            is_likely_special_block = (len(paragraph) > 2 and len(paragraph_content) / len(paragraph) < 20)
            if len(paragraph_content) < 30 and not is_likely_special_block:
                continue
                
            # Check if this paragraph is a special block (code block, table, etc.)
            is_special_block = False
            for block_start, block_end in special_blocks:
                if relative_start == block_start and relative_end == block_end:
                    is_special_block = True
                    break
                    
            # Special blocks should be kept intact
            if is_special_block:
                block_type = "Block"
                if paragraph_content.strip().startswith("```") or paragraph_content.strip().startswith("~~~"):
                    block_type = "Code Block"
                elif "|" in paragraph_content and not paragraph_content.startswith(">"):
                    block_type = "Table"
                elif paragraph_content.strip().startswith(">"):
                    block_type = "Blockquote"
                elif any(paragraph_content.lstrip().startswith(marker + " ") for marker in ["-", "*", "+"]) or re.match(r"^\d+\.\s", paragraph_content.lstrip()):
                    block_type = "List"
                    
                section_title = f"{title} - {block_type}" if title else f"{block_type}"
                sections.append(
                    Section(
                        title=section_title,
                        content=paragraph_content,
                        start_idx=absolute_start,
                        end_idx=absolute_end
                    )
                )
                continue
                
            
        # Create one section for the paragraph
        section_title = f"{title} - Par. {p_idx+1}" if title else f"Paragraph {p_idx+1}"
        sections.append(
            Section(
                title=section_title,
                content=paragraph_content,
                start_idx=absolute_start,
                end_idx=absolute_end
            )
        )
                
        # If no sections were created (e.g., all paragraphs were too small),
        # create one section for the entire content
        if not sections and text.strip():
            sections = [
                Section(
                    title=title,
                    content=text,
                    start_idx=start_idx,
                    end_idx=end_idx
                )
            ]
            
        return sections
    
    def extract_hierarchical_sections(self, text: str, max_section_length: int = 1000) -> List[Section]:
        """
        Extract sections from text by combining header-based sections with fine-grained paragraph sections.
        This approach keeps the hierarchical structure but adds additional granularity within each section.
        Special markdown elements like tables, code blocks, blockquotes, and lists are kept intact.

        Args:
            text (str): The text to segment
            max_section_length (int): Maximum length for fine-grained sections

        Returns:
            List[Section]: Combined list of sections with varying granularity

        Example:
            >>> parser = MarkdownParser()
            >>> sections = parser.extract_hierarchical_sections("# Header\\nLong content here.\\n## Subheader\\nMore details.")
            >>> len(sections) > 0
            True
        """
        content_map: Dict[str, Section] = {} # Use content hash as key, section as value

        # Helper function to add sections while handling duplicates
        def add_or_update_section(section: Section):
            if section.title == "" or section.content == "":
                return
            
            # Calculate hash of the content
            content_hash = hashlib.sha256(section.content.encode('utf-8')).hexdigest()

            if content_hash not in content_map or \
               len(section.title) < len(content_map[content_hash].title):
                content_map[content_hash] = section

        # First get the header-based sections
        header_sections = self.extract_sections(text)
        
        # Process header sections and their corresponding fine-grained sections
        for section in header_sections:
            # Consider the original header section
            add_or_update_section(section)
            
            # Create fine-grained sections from this section's content
            fine_sections = self.extract_fine_grained_sections(
                section.content,
                title=section.title,
                start_idx=section.start_idx,
                end_idx=section.end_idx,
                max_section_length=max_section_length
            )
            
            # Consider the fine-grained sections
            for fine_section in fine_sections:
                add_or_update_section(fine_section)
            
        # If no header sections were found, process the entire text as one block
        if not header_sections and text.strip():
            fine_sections = self.extract_fine_grained_sections(
                text,
                title="",  # No title if no headers
                start_idx=0,
                end_idx=len(text.split("\n")) -1,
                max_section_length=max_section_length
            )
            for fine_section in fine_sections:
                add_or_update_section(fine_section)

        # Return the unique sections stored in the map values
        return list(content_map.values())
