import pytest
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from mlentory_extract.core.MarkdownParser import MarkdownParser, Section


@pytest.fixture
def parser():
    """Create and return a MarkdownParser instance."""
    return MarkdownParser()


def test_empty_document(parser):
    """Test that empty documents are handled properly."""
    empty_text = ""
    sections = parser.extract_sections(empty_text)
    
    assert len(sections) == 0, "Empty document should yield no sections"
    
    # Test hierarchical sections too
    hierarchical_sections = parser.extract_hierarchical_sections(empty_text)
    assert len(hierarchical_sections) == 0, "Empty document should yield no hierarchical sections"


def test_document_without_headers(parser):
    """Test that documents without headers create one section with the entire content."""
    text_without_headers = "This is a document without any headers.\nIt should be treated as one section."
    sections = parser.extract_sections(text_without_headers)
    
    assert len(sections) == 1, "Document without headers should yield one section"
    assert sections[0].title == "", "Section title should be empty"
    assert "without any headers" in sections[0].content, "Section should contain the document content"


def test_basic_headers(parser):
    """Test extraction of basic header structures."""
    md_text = """# Header 1
This is content under header 1.

## Header 1.1
This is content under header 1.1.

# Header 2
This is content under header 2."""

    sections = parser.extract_sections(md_text)
    
    assert len(sections) == 3, "Should identify 3 distinct header sections"
    assert sections[0].title == "Header 1", "First section title should be 'Header 1'"
    assert sections[1].title == "Header 1 > Header 1.1", "Second section should have hierarchical title"
    assert sections[2].title == "Header 2", "Third section title should be 'Header 2'"


def test_code_blocks(parser):
    """Test handling of code blocks with # that should not be treated as headers."""
    md_text = """# Header 1
Here's some Python code:

```python
# This is a comment, not a header
def hello():
    print("Hello, world!")
```

And here's a comment in some other code:

~~~
# Also not a header
let x = 42;
~~~

## Header 1.1
Content after code blocks."""

    sections = parser.extract_sections(md_text)
    
    # Verify we got 2 sections (Header 1 and Header 1.1)
    assert len(sections) == 2, "Should identify 2 header sections, ignoring # in code blocks"
    
    # Code blocks should be included in their parent section content
    assert "```python" in sections[0].content, "Code block should be in section content"
    assert "# This is a comment" in sections[0].content, "Comment in code block should be preserved"
    assert "~~~" in sections[0].content, "Alternative code block marker should be preserved"


def test_nested_headers(parser):
    """Test extraction of deeply nested header structures."""
    md_text = """# Level 1
Content level 1.

## Level 2
Content level 2.

### Level 3
Content level 3.

#### Level 4
Content level 4.

## Another Level 2
Back to level 2."""

    sections = parser.extract_sections(md_text)
    
    assert len(sections) == 5, "Should identify 5 distinct header sections"
    assert sections[2].title == "Level 1 > Level 2 > Level 3", "Should build correct hierarchical title"
    assert sections[3].title == "Level 1 > Level 2 > Level 3 > Level 4", "Should handle deep nesting"
    assert sections[4].title == "Level 1 > Another Level 2", "Should reset hierarchy correctly"


def test_tables(parser):
    """Test handling of markdown tables."""
    md_text = """# Section with Table
Here's a table:

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |

More content after the table."""

    # First test that regular section extraction works
    sections = parser.extract_sections(md_text)
    assert len(sections) == 1, "Should have one section with the table"
    assert "| Column 1 |" in sections[0].content, "Table should be preserved in the section"
    
    # Now test that fine-grained extraction keeps the table intact
    fine_sections = parser.extract_fine_grained_sections(md_text)
    
    # Find the table section
    table_section = None
    for section in fine_sections:
        if "Table" in section.title:
            table_section = section
            break
    
    assert table_section is not None, "Table should be identified as a special block"
    assert "| Column 1 |" in table_section.content, "Table content should be preserved"
    assert "| Value 4  |" in table_section.content, "Table data should be preserved"


def test_blockquotes(parser):
    """Test handling of blockquotes."""
    md_text = """# Section with Blockquote
Here's a blockquote:

> This is a blockquote.
> It can span multiple lines.
> 
> And can have paragraphs.

More content after the blockquote."""

    # Test that fine-grained extraction keeps blockquotes intact
    fine_sections = parser.extract_fine_grained_sections(md_text)
    
    # Find the blockquote section
    blockquote_section = None
    for section in fine_sections:
        if "Blockquote" in section.title:
            blockquote_section = section
            break
    
    assert blockquote_section is not None, "Blockquote should be identified as a special block"
    assert "> This is a blockquote." in blockquote_section.content, "Blockquote content should be preserved"
    assert "> And can have paragraphs." in blockquote_section.content, "Blockquote paragraphs should be preserved"


def test_lists(parser):
    """Test handling of lists (unordered and ordered)."""
    md_text = """# Section with Lists
Here's an unordered list:

- Item 1
- Item 2
  - Subitem 2.1
  - Subitem 2.2
- Item 3

And an ordered list:

1. First item
2. Second item
   1. Subitem 2.1
   2. Subitem 2.2
3. Third item

More content after the lists."""

    # Test that fine-grained extraction keeps lists intact
    fine_sections = parser.extract_fine_grained_sections(md_text)
    
    # Find the list sections
    unordered_list_section = None
    ordered_list_section = None
    
    print("FINE SECTIONS: ")
    
    for section in fine_sections:
        print(section.title)
        print(section.content)
        print("--------------------------------")
        
    for section in fine_sections:
        if "List" in section.title:
            if "- Item 1" in section.content:
                unordered_list_section = section
            elif "1. First item" in section.content:
                ordered_list_section = section
    
    assert unordered_list_section is not None, "Unordered list should be identified as a special block"
    assert ordered_list_section is not None, "Ordered list should be identified as a special block"
    
    assert "- Item 1" in unordered_list_section.content, "Unordered list items should be preserved"
    assert "  - Subitem 2.1" in unordered_list_section.content, "Nested list items should be preserved"
    
    assert "1. First item" in ordered_list_section.content, "Ordered list items should be preserved"
    assert "   1. Subitem 2.1" in ordered_list_section.content, "Nested ordered list items should be preserved"


def test_mixed_content(parser):
    """Test handling of documents with mixed content types."""
    md_text = """# Mixed Content Section
This section has a bit of everything.

## Code
```python
# A comment in code
def example():
    return "Hello"
```

## Table
| Name | Value |
|------|-------|
| Test | 123   |

## List and Blockquote
- Item with > not a blockquote
- Normal item

> A blockquote
> With multiple lines

Normal paragraph here.

## Conclusion
That's all!"""

    # Test hierarchical section extraction with mixed content
    hierarchical_sections = parser.extract_hierarchical_sections(md_text)
    
    # We should have a mix of header sections and special block sections
    assert len(hierarchical_sections) > 4, "Should identify multiple sections with mixed content"
    
    # Verify each special block was properly identified
    special_blocks = {
        "Code Block": False,
        "Table": False,
        "List": False,
        "Blockquote": False
    }
    
    for section in hierarchical_sections:
        for block_type in special_blocks:
            if block_type in section.title:
                special_blocks[block_type] = True
    
    for block_type, found in special_blocks.items():
        assert found, f"{block_type} should be identified as a special block"


@pytest.mark.parametrize("lines,start_idx,expected_is_special,expected_end_idx", [
    (["```python", "code", "```"], 0, True, 2),  # Code block
    (["|col1|col2|", "|---|---|", "|val1|val2|"], 0, True, 2),  # Table
    (["> blockquote", "> line 2"], 0, True, 1),  # Blockquote
    (["- item1", "- item2", "  - subitem"], 0, True, 2),  # List
    (["1. item1", "2. item2"], 0, True, 1),  # Numbered list
    (["Regular text", "not special"], 0, False, 0),  # Not special
])
def test_special_markdown_block_detection(parser, lines, start_idx, expected_is_special, expected_end_idx):
    """Test the _is_special_markdown_block method directly with parametrized test cases."""
    is_special, end_idx = parser._is_special_markdown_block(lines, start_idx)
    assert is_special == expected_is_special, f"Failed for case: {lines}"
    if expected_is_special:
        assert end_idx == expected_end_idx, f"Wrong end_idx for case: {lines}"


def test_fine_grained_section_extraction(parser):
    """Test extraction of fine-grained sections."""
    md_text = """This is a paragraph with several sentences. It should be kept together.
It contains multiple lines that form one logical paragraph.\n

This is another paragraph. It should be separate from the first.
It also has multiple sentences to test sentence-based splitting if needed.\n

A short line.\n

A very long paragraph that should be split into sentences if it exceeds the maximum length. This sentence adds more content to ensure that the paragraph becomes long enough to trigger the sentence splitting logic. We need to have enough text to exceed the threshold. More text is added to make this paragraph even longer and ensure it will be split into multiple chunks based on sentence boundaries.\n"""

    # Test with a small max_section_length to force splitting
    fine_sections = parser.extract_fine_grained_sections(md_text, max_section_length=20)
    
    print("FINE Grained Sections: ", len(fine_sections))
    
    for section in fine_sections:
        print(section.title)
        print(section.content)
        print("--------------------------------")
    
    # We should have more sections than paragraphs due to splitting
    assert len(fine_sections) > 3, "Long paragraphs should be split into smaller sections"
    
    # Test with a large max_section_length to avoid splitting
    large_sections = parser.extract_fine_grained_sections(md_text, max_section_length=1000)
    
    # We should get roughly one section per paragraph
    assert len(large_sections) <= 4, "With large max length, should have one section per paragraph"


def test_hierarchical_section_extraction(parser):
    """Test extraction of hierarchical sections."""
    md_text = """# Main Header
This is the main content.

## Subheader 1
This is subcontent 1.

## Subheader 2
This is subcontent 2.

### Nested Subheader
This is nested content."""

    hierarchical_sections = parser.extract_hierarchical_sections(md_text)
    
    # We should have at least the header sections
    assert len(hierarchical_sections) >= 4, "Should have all header sections"
   
    # Verify the hierarchical structure is preserved
    header_titles = [section.title for section in hierarchical_sections if " - Par." not in section.title]
    
    assert "Main Header" in header_titles
    assert "Main Header > Subheader 1" in header_titles
    assert "Main Header > Subheader 2" in header_titles
    assert "Main Header > Subheader 2 > Nested Subheader" in header_titles 