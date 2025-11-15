"""
AST-Based Markdown Parser - Phase 1

Bu modül markdown dosyalarını AST (Abstract Syntax Tree) kullanarak intelligent 
parsing yapar ve semantic chunking için optimize edilmiş structure extraction sağlar.

Features:
- Header hierarchy awareness (H1→H6 relationships)
- Table preservation with semantic context
- Code block intelligent handling
- Math formula protection
- Cross-reference resolution
- Turkish language optimizations
- Performance caching

AST-based yaklaşım sayesinde traditional regex-based parsing'den çok daha güvenilir
ve semantic-aware markdown processing yapılır.
"""

import re
import ast
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path

# Core dependencies
try:
    # Try relative imports first (when used as package)
    from ..config import get_config
    from ..utils.logger import get_logger
except ImportError:
    # Fallback to absolute imports (for testing and standalone use)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.config import get_config
    from src.utils.logger import get_logger

# Enhanced dependencies for Phase 1
try:
    import markdown
    from bs4 import BeautifulSoup, Tag, NavigableString
    import lxml
    from cachetools import LRUCache
    
    # Try to import markdown extensions - some may not be available
    try:
        from markdown.extensions import codehilite, tables, toc
        from markdown.preprocessors import Preprocessor
        from markdown.postprocessors import Postprocessor
        from markdown.treeprocessors import Treeprocessor
        FULL_MARKDOWN_EXTENSIONS = True
    except ImportError:
        FULL_MARKDOWN_EXTENSIONS = False
    
    # Math extension is optional
    try:
        from markdown.extensions import math
        MATH_EXTENSION_AVAILABLE = True
    except ImportError:
        MATH_EXTENSION_AVAILABLE = False
    
    import xml.etree.ElementTree as ET
    MARKDOWN_SUPPORT = True
    
except ImportError as e:
    MARKDOWN_SUPPORT = False
    FULL_MARKDOWN_EXTENSIONS = False
    MATH_EXTENSION_AVAILABLE = False
    
    # Fallback definitions for when BeautifulSoup is not available
    class Tag:
        def __init__(self):
            self.name = ""
            self.attrs = {}
        def get_text(self):
            return ""
        def find(self, *args, **kwargs):
            return None
        def find_all(self, *args, **kwargs):
            return []
    
    class NavigableString:
        def __init__(self, text=""):
            self.text = text
        def strip(self):
            return self.text.strip()
    
    class BeautifulSoup:
        def __init__(self, *args, **kwargs):
            self.children = []
    
    print(f"Warning: Advanced markdown parsing features not available: {e}")


class MarkdownElementType(Enum):
    """Markdown element türleri."""
    HEADER = "header"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    CODE_BLOCK = "code_block"
    INLINE_CODE = "inline_code"
    TABLE = "table"
    MATH_BLOCK = "math_block"
    MATH_INLINE = "math_inline"
    QUOTE = "quote"
    LINK = "link"
    IMAGE = "image"
    HORIZONTAL_RULE = "horizontal_rule"
    TEXT = "text"
    HTML = "html"


@dataclass
class MarkdownElement:
    """AST node for markdown elements."""
    element_type: MarkdownElementType
    content: str
    level: Optional[int] = None  # For headers (1-6), lists (nesting level)
    attributes: Optional[Dict[str, Any]] = None
    children: Optional[List['MarkdownElement']] = None
    parent: Optional['MarkdownElement'] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.attributes is None:
            self.attributes = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MarkdownSection:
    """Semantic section for chunking."""
    title: str
    level: int
    content: str
    elements: List[MarkdownElement]
    start_line: int
    end_line: int
    subsections: List['MarkdownSection']
    context: Dict[str, Any]
    
    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []
        if self.context is None:
            self.context = {}


@dataclass
class TableContext:
    """Table semantic context."""
    headers: List[str]
    row_count: int
    column_count: int
    table_type: str  # "data", "comparison", "summary", etc.
    semantic_summary: str


@dataclass
class CodeContext:
    """Code block semantic context."""
    language: Optional[str]
    code_type: str  # "example", "function", "config", etc.
    complexity_level: str  # "simple", "intermediate", "complex"
    description: Optional[str]


class ASTMarkdownParser:
    """
    Advanced AST-based Markdown Parser with semantic understanding.
    
    Bu parser markdown content'i semantic chunks için optimize edilmiş
    abstract syntax tree'ye dönüştürür.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__, self.config)
        
        # Caching for performance
        self.parse_cache = LRUCache(maxsize=100)
        self.element_cache = LRUCache(maxsize=500)
        
        # Turkish language specific patterns
        self.turkish_header_patterns = re.compile(r'\b(Bölüm|Kısım|Başlık|Konu|Giriş|Sonuç|Özet|Kaynaklar)\b', re.IGNORECASE)
        self.turkish_list_indicators = re.compile(r'^\s*[-\*\+\•]|\d+[\.\)]\s+|[a-zA-Z][\.\)]\s+', re.MULTILINE)
        
        # Math formula patterns
        self.math_patterns = {
            'block': re.compile(r'^\s*\$\$([^$]+?)\$\$\s*$', re.MULTILINE | re.DOTALL),
            'inline': re.compile(r'\$([^$]+?)\$'),
            'latex_block': re.compile(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', re.DOTALL),
            'latex_inline': re.compile(r'\\[a-zA-Z]+(?:\{[^}]*\})*')
        }
        
        # Code block patterns
        self.code_patterns = {
            'fenced': re.compile(r'^```([^\n]*)\n(.*?)^```', re.MULTILINE | re.DOTALL),
            'indented': re.compile(r'^(?: {4}|\t)(.*)$', re.MULTILINE),
            'inline': re.compile(r'`([^`]+)`')
        }
        
        # Reference patterns
        self.reference_patterns = {
            'link_ref': re.compile(r'\[([^\]]+)\]\[([^\]]*)\]'),
            'link_direct': re.compile(r'\[([^\]]+)\]\(([^)]+)\)'),
            'footnote': re.compile(r'\[\^([^\]]+)\]'),
            'internal_link': re.compile(r'\[\[([^\]]+)\]\]')
        }
        
        # Initialize markdown processor
        if MARKDOWN_SUPPORT:
            self._initialize_markdown_processor()
        else:
            self.logger.warning("Falling back to basic markdown parsing without full AST support")
            
    def _initialize_markdown_processor(self):
        """Initialize markdown processor with extensions."""
        try:
            # Build list of available extensions
            extensions = [
                'markdown.extensions.tables',
                'markdown.extensions.fenced_code',
                'markdown.extensions.footnotes',
                'markdown.extensions.attr_list'
            ]
            
            extension_configs = {}
            
            # Add optional extensions if available
            if FULL_MARKDOWN_EXTENSIONS:
                extensions.extend([
                    'markdown.extensions.codehilite',
                    'markdown.extensions.toc'
                ])
                extension_configs.update({
                    'markdown.extensions.codehilite': {
                        'css_class': 'highlight',
                        'use_pygments': False
                    },
                    'markdown.extensions.toc': {
                        'anchorlink': True,
                        'permalink': True
                    }
                })
            
            if MATH_EXTENSION_AVAILABLE:
                extensions.append('markdown.extensions.math')
            
            self.md_processor = markdown.Markdown(
                extensions=extensions,
                extension_configs=extension_configs
            )
            self.logger.info(f"Markdown processor initialized with {len(extensions)} extensions")
        except Exception as e:
            self.logger.error(f"Failed to initialize markdown processor: {e}")
            self.md_processor = None
    
    def parse_markdown_to_ast(self, markdown_text: str) -> List[MarkdownElement]:
        """
        Parse markdown text into Abstract Syntax Tree.
        
        Args:
            markdown_text: Raw markdown content
            
        Returns:
            List of MarkdownElement nodes representing the AST
        """
        if not markdown_text.strip():
            return []
        
        # Check cache
        cache_key = hashlib.md5(markdown_text.encode()).hexdigest()
        if cache_key in self.parse_cache:
            self.logger.debug("Cache hit for AST parsing")
            return self.parse_cache[cache_key]
        
        self.logger.info(f"Parsing markdown to AST (length: {len(markdown_text)})")
        
        try:
            # Step 1: Pre-process and protect special elements
            protected_text, protection_map = self._protect_special_elements(markdown_text)
            
            # Step 2: Parse with markdown processor if available
            if MARKDOWN_SUPPORT and self.md_processor:
                ast_nodes = self._parse_with_markdown_processor(protected_text, protection_map)
            else:
                # Fallback to manual parsing
                ast_nodes = self._manual_parse_markdown(protected_text, protection_map)
            
            # Step 3: Post-process and build hierarchy
            structured_ast = self._build_hierarchical_structure(ast_nodes)
            
            # Step 4: Add semantic context
            enhanced_ast = self._add_semantic_context(structured_ast)
            
            # Cache result
            self.parse_cache[cache_key] = enhanced_ast
            
            self.logger.info(f"Generated AST with {len(enhanced_ast)} top-level nodes")
            return enhanced_ast
            
        except Exception as e:
            self.logger.error(f"AST parsing failed: {e}")
            # Fallback to basic text parsing
            return self._fallback_text_parsing(markdown_text)
    
    def _protect_special_elements(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Protect special elements (math, code) from markdown processing.
        
        Returns:
            Tuple of (protected_text, protection_map)
        """
        protection_map = {}
        protected_text = text
        
        # Protect math blocks
        for match in self.math_patterns['block'].finditer(text):
            placeholder = f"__MATH_BLOCK_{len(protection_map)}__"
            protection_map[placeholder] = match.group(0)
            protected_text = protected_text.replace(match.group(0), placeholder)
        
        # Protect code blocks
        for match in self.code_patterns['fenced'].finditer(text):
            placeholder = f"__CODE_BLOCK_{len(protection_map)}__"
            protection_map[placeholder] = match.group(0)
            protected_text = protected_text.replace(match.group(0), placeholder)
        
        # Protect inline math
        for match in self.math_patterns['inline'].finditer(protected_text):
            placeholder = f"__MATH_INLINE_{len(protection_map)}__"
            protection_map[placeholder] = match.group(0)
            protected_text = protected_text.replace(match.group(0), placeholder)
        
        # Protect inline code
        for match in self.code_patterns['inline'].finditer(protected_text):
            placeholder = f"__CODE_INLINE_{len(protection_map)}__"
            protection_map[placeholder] = match.group(0)
            protected_text = protected_text.replace(match.group(0), placeholder)
        
        return protected_text, protection_map
    
    def _parse_with_markdown_processor(self, text: str, protection_map: Dict[str, str]) -> List[MarkdownElement]:
        """Parse using markdown processor and convert to AST."""
        try:
            # Convert to HTML
            html = self.md_processor.convert(text)
            
            # Parse HTML to AST
            soup = BeautifulSoup(html, 'html.parser')
            ast_nodes = []
            
            for element in soup.children:
                if isinstance(element, Tag):
                    node = self._html_element_to_ast(element, protection_map)
                    if node:
                        ast_nodes.append(node)
                elif isinstance(element, NavigableString) and element.strip():
                    # Handle text nodes
                    text_node = MarkdownElement(
                        element_type=MarkdownElementType.TEXT,
                        content=str(element).strip()
                    )
                    ast_nodes.append(text_node)
            
            return ast_nodes
            
        except Exception as e:
            self.logger.error(f"Markdown processor parsing failed: {e}")
            return self._manual_parse_markdown(text, protection_map)
    
    def _html_element_to_ast(self, element: Tag, protection_map: Dict[str, str]) -> Optional[MarkdownElement]:
        """Convert HTML element to MarkdownElement."""
        tag_name = element.name.lower()
        
        # Headers
        if tag_name.startswith('h') and len(tag_name) == 2 and tag_name[1].isdigit():
            level = int(tag_name[1])
            content = self._restore_protected_content(element.get_text(), protection_map)
            return MarkdownElement(
                element_type=MarkdownElementType.HEADER,
                content=content,
                level=level,
                attributes={'id': element.get('id', '')}
            )
        
        # Paragraphs
        elif tag_name == 'p':
            content = self._restore_protected_content(element.get_text(), protection_map)
            return MarkdownElement(
                element_type=MarkdownElementType.PARAGRAPH,
                content=content
            )
        
        # Lists
        elif tag_name in ['ul', 'ol']:
            list_node = MarkdownElement(
                element_type=MarkdownElementType.LIST,
                content="",
                attributes={'ordered': tag_name == 'ol'}
            )
            
            for li in element.find_all('li', recursive=False):
                li_content = self._restore_protected_content(li.get_text(), protection_map)
                list_item = MarkdownElement(
                    element_type=MarkdownElementType.LIST_ITEM,
                    content=li_content,
                    parent=list_node
                )
                list_node.children.append(list_item)
            
            return list_node
        
        # Tables
        elif tag_name == 'table':
            return self._parse_table_element(element, protection_map)
        
        # Code blocks
        elif tag_name == 'pre':
            code_element = element.find('code')
            if code_element:
                content = self._restore_protected_content(code_element.get_text(), protection_map)
                language = None
                
                # Extract language from class
                if 'class' in code_element.attrs:
                    classes = code_element['class']
                    for cls in classes:
                        if cls.startswith('language-'):
                            language = cls[9:]  # Remove 'language-' prefix
                            break
                
                return MarkdownElement(
                    element_type=MarkdownElementType.CODE_BLOCK,
                    content=content,
                    attributes={'language': language}
                )
        
        # Blockquotes
        elif tag_name == 'blockquote':
            content = self._restore_protected_content(element.get_text(), protection_map)
            return MarkdownElement(
                element_type=MarkdownElementType.QUOTE,
                content=content
            )
        
        # Divs and other containers
        elif tag_name == 'div':
            # Handle special div types (math blocks, etc.)
            if 'class' in element.attrs:
                classes = element['class']
                if 'math' in classes:
                    content = self._restore_protected_content(element.get_text(), protection_map)
                    return MarkdownElement(
                        element_type=MarkdownElementType.MATH_BLOCK,
                        content=content
                    )
            
            # Generic container - process children
            container_elements = []
            for child in element.children:
                if isinstance(child, Tag):
                    child_node = self._html_element_to_ast(child, protection_map)
                    if child_node:
                        container_elements.append(child_node)
            
            if container_elements:
                # Return first child if only one, otherwise wrap in HTML element
                if len(container_elements) == 1:
                    return container_elements[0]
                else:
                    container = MarkdownElement(
                        element_type=MarkdownElementType.HTML,
                        content="",
                        children=container_elements
                    )
                    return container
        
        # Horizontal rules
        elif tag_name == 'hr':
            return MarkdownElement(
                element_type=MarkdownElementType.HORIZONTAL_RULE,
                content="---"
            )
        
        return None
    
    def _parse_table_element(self, table: Tag, protection_map: Dict[str, str]) -> MarkdownElement:
        """Parse HTML table to MarkdownElement with semantic context."""
        headers = []
        rows = []
        
        # Extract headers
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [self._restore_protected_content(th.get_text().strip(), protection_map) 
                          for th in header_row.find_all(['th', 'td'])]
        
        # Extract rows
        tbody = table.find('tbody') or table
        for tr in tbody.find_all('tr'):
            if tr.parent.name != 'thead':  # Skip header rows
                row_data = [self._restore_protected_content(td.get_text().strip(), protection_map) 
                           for td in tr.find_all(['td', 'th'])]
                if row_data:
                    rows.append(row_data)
        
        # Create table context
        table_context = TableContext(
            headers=headers,
            row_count=len(rows),
            column_count=len(headers) if headers else (len(rows[0]) if rows else 0),
            table_type=self._determine_table_type(headers, rows),
            semantic_summary=self._generate_table_summary(headers, rows)
        )
        
        # Format table content
        table_content = self._format_table_content(headers, rows)
        
        return MarkdownElement(
            element_type=MarkdownElementType.TABLE,
            content=table_content,
            metadata={'table_context': table_context}
        )
    
    def _manual_parse_markdown(self, text: str, protection_map: Dict[str, str]) -> List[MarkdownElement]:
        """Manual parsing as fallback when markdown processor is not available."""
        ast_nodes = []
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            # Skip empty lines
            if not line.strip():
                i += 1
                continue
            
            # Headers
            if line.startswith('#'):
                level = 0
                while level < len(line) and line[level] == '#':
                    level += 1
                
                if level <= 6 and (level == len(line) or line[level] == ' '):
                    content = line[level:].strip()
                    content = self._restore_protected_content(content, protection_map)
                    
                    header_node = MarkdownElement(
                        element_type=MarkdownElementType.HEADER,
                        content=content,
                        level=level,
                        start_line=i
                    )
                    ast_nodes.append(header_node)
                    i += 1
                    continue
            
            # Horizontal rules
            if re.match(r'^[-*_]{3,}\s*$', line):
                hr_node = MarkdownElement(
                    element_type=MarkdownElementType.HORIZONTAL_RULE,
                    content=line,
                    start_line=i
                )
                ast_nodes.append(hr_node)
                i += 1
                continue
            
            # Lists
            if re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
                list_node, next_i = self._parse_list_block(lines, i, protection_map)
                ast_nodes.append(list_node)
                i = next_i
                continue
            
            # Blockquotes
            if line.startswith('>'):
                quote_node, next_i = self._parse_quote_block(lines, i, protection_map)
                ast_nodes.append(quote_node)
                i = next_i
                continue
            
            # Code blocks (indented)
            if line.startswith('    ') or line.startswith('\t'):
                code_node, next_i = self._parse_indented_code_block(lines, i, protection_map)
                ast_nodes.append(code_node)
                i = next_i
                continue
            
            # Regular paragraphs
            paragraph_node, next_i = self._parse_paragraph_block(lines, i, protection_map)
            ast_nodes.append(paragraph_node)
            i = next_i
        
        return ast_nodes
    
    def _parse_list_block(self, lines: List[str], start_idx: int, protection_map: Dict[str, str]) -> Tuple[MarkdownElement, int]:
        """Parse list block manually."""
        first_line = lines[start_idx].strip()
        is_ordered = re.match(r'^\s*\d+\.\s+', first_line) is not None
        
        list_node = MarkdownElement(
            element_type=MarkdownElementType.LIST,
            content="",
            attributes={'ordered': is_ordered},
            start_line=start_idx
        )
        
        i = start_idx
        while i < len(lines):
            line = lines[i].rstrip()
            
            # Empty line
            if not line.strip():
                i += 1
                continue
            
            # List item
            if (is_ordered and re.match(r'^\s*\d+\.\s+', line)) or \
               (not is_ordered and re.match(r'^\s*[-*+]\s+', line)):
                
                # Extract item content
                content_match = re.match(r'^\s*(?:\d+\.|\*|\+|\-)\s*(.*)', line)
                item_content = content_match.group(1) if content_match else ""
                
                # Collect continuation lines
                i += 1
                while i < len(lines):
                    next_line = lines[i].rstrip()
                    if not next_line.strip():
                        i += 1
                        continue
                    
                    if re.match(r'^\s*(?:\d+\.|\*|\+|\-)\s+', next_line):
                        break  # Next list item
                    
                    if next_line.startswith('  ') or next_line.startswith('\t'):
                        item_content += " " + next_line.strip()
                        i += 1
                    else:
                        break  # End of list
                
                item_content = self._restore_protected_content(item_content, protection_map)
                
                list_item = MarkdownElement(
                    element_type=MarkdownElementType.LIST_ITEM,
                    content=item_content,
                    parent=list_node
                )
                list_node.children.append(list_item)
            else:
                break
        
        list_node.end_line = i - 1
        return list_node, i
    
    def _parse_quote_block(self, lines: List[str], start_idx: int, protection_map: Dict[str, str]) -> Tuple[MarkdownElement, int]:
        """Parse blockquote manually."""
        quote_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            if line.startswith('>'):
                # Remove quote marker and leading space
                content = line[1:].lstrip()
                quote_lines.append(content)
                i += 1
            elif not line.strip() and i < len(lines) - 1 and lines[i + 1].startswith('>'):
                # Empty line within quote
                quote_lines.append("")
                i += 1
            else:
                break
        
        quote_content = '\n'.join(quote_lines)
        quote_content = self._restore_protected_content(quote_content, protection_map)
        
        quote_node = MarkdownElement(
            element_type=MarkdownElementType.QUOTE,
            content=quote_content,
            start_line=start_idx,
            end_line=i - 1
        )
        
        return quote_node, i
    
    def _parse_indented_code_block(self, lines: List[str], start_idx: int, protection_map: Dict[str, str]) -> Tuple[MarkdownElement, int]:
        """Parse indented code block."""
        code_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            if line.startswith('    '):
                code_lines.append(line[4:])  # Remove 4 spaces
                i += 1
            elif line.startswith('\t'):
                code_lines.append(line[1:])  # Remove tab
                i += 1
            elif not line.strip():
                # Empty line within code block
                code_lines.append("")
                i += 1
            else:
                break
        
        code_content = '\n'.join(code_lines).rstrip()
        code_content = self._restore_protected_content(code_content, protection_map)
        
        code_node = MarkdownElement(
            element_type=MarkdownElementType.CODE_BLOCK,
            content=code_content,
            start_line=start_idx,
            end_line=i - 1
        )
        
        return code_node, i
    
    def _parse_paragraph_block(self, lines: List[str], start_idx: int, protection_map: Dict[str, str]) -> Tuple[MarkdownElement, int]:
        """Parse paragraph block."""
        para_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            # Empty line ends paragraph
            if not line.strip():
                break
            
            # Special markdown elements end paragraph
            if (line.startswith('#') or 
                line.startswith('>') or
                line.startswith('    ') or
                line.startswith('\t') or
                re.match(r'^\s*[-*+]\s+', line) or
                re.match(r'^\s*\d+\.\s+', line) or
                re.match(r'^[-*_]{3,}\s*$', line)):
                break
            
            para_lines.append(line)
            i += 1
        
        para_content = ' '.join(para_lines)
        para_content = self._restore_protected_content(para_content, protection_map)
        
        para_node = MarkdownElement(
            element_type=MarkdownElementType.PARAGRAPH,
            content=para_content,
            start_line=start_idx,
            end_line=i - 1
        )
        
        return para_node, i
    
    def _restore_protected_content(self, text: str, protection_map: Dict[str, str]) -> str:
        """Restore protected content (math, code, etc.)."""
        restored = text
        for placeholder, original in protection_map.items():
            restored = restored.replace(placeholder, original)
        return restored
    
    def _build_hierarchical_structure(self, ast_nodes: List[MarkdownElement]) -> List[MarkdownElement]:
        """Build hierarchical structure based on headers."""
        if not ast_nodes:
            return []
        
        # Find header nodes and build hierarchy
        structured_nodes = []
        header_stack = []  # Stack to track header hierarchy
        
        for node in ast_nodes:
            if node.element_type == MarkdownElementType.HEADER:
                # Close any headers at same or lower levels
                while header_stack and header_stack[-1].level >= node.level:
                    header_stack.pop()
                
                # Set parent relationship
                if header_stack:
                    node.parent = header_stack[-1]
                    header_stack[-1].children.append(node)
                else:
                    structured_nodes.append(node)
                
                header_stack.append(node)
            
            else:
                # Non-header node
                if header_stack:
                    # Add to current header section
                    node.parent = header_stack[-1]
                    header_stack[-1].children.append(node)
                else:
                    # Top-level content
                    structured_nodes.append(node)
        
        return structured_nodes
    
    def _add_semantic_context(self, ast_nodes: List[MarkdownElement]) -> List[MarkdownElement]:
        """Add semantic context to AST nodes."""
        for node in ast_nodes:
            self._enhance_node_semantics(node)
            # Recursively process children
            if node.children:
                self._add_semantic_context(node.children)
        
        return ast_nodes
    
    def _enhance_node_semantics(self, node: MarkdownElement):
        """Enhance a single node with semantic information."""
        
        # Header semantic analysis
        if node.element_type == MarkdownElementType.HEADER:
            node.metadata['semantic_type'] = self._classify_header_type(node.content)
            node.metadata['keywords'] = self._extract_header_keywords(node.content)
        
        # Code block analysis
        elif node.element_type == MarkdownElementType.CODE_BLOCK:
            language = node.attributes.get('language', '')
            code_context = self._analyze_code_block(node.content, language)
            node.metadata['code_context'] = code_context
        
        # Table analysis
        elif node.element_type == MarkdownElementType.TABLE:
            # Table context already added during parsing
            pass
        
        # List analysis
        elif node.element_type == MarkdownElementType.LIST:
            node.metadata['list_type'] = self._classify_list_type(node)
            node.metadata['item_count'] = len(node.children)
        
        # Paragraph analysis
        elif node.element_type == MarkdownElementType.PARAGRAPH:
            node.metadata['paragraph_type'] = self._classify_paragraph_type(node.content)
            node.metadata['sentence_count'] = len(re.split(r'[.!?]+', node.content))
    
    def _classify_header_type(self, content: str) -> str:
        """Classify header type based on content."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['giriş', 'introduction', 'başlangıç']):
            return 'introduction'
        elif any(word in content_lower for word in ['sonuç', 'conclusion', 'özet', 'summary']):
            return 'conclusion'
        elif any(word in content_lower for word in ['yöntem', 'method', 'metodoloji']):
            return 'methodology'
        elif any(word in content_lower for word in ['bulgular', 'results', 'sonuçlar']):
            return 'results'
        elif any(word in content_lower for word in ['tartışma', 'discussion', 'değerlendirme']):
            return 'discussion'
        elif any(word in content_lower for word in ['kaynaklar', 'references', 'bibliyografya']):
            return 'references'
        else:
            return 'general'
    
    def _extract_header_keywords(self, content: str) -> List[str]:
        """Extract keywords from header content."""
        # Remove common words and extract meaningful terms
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Turkish stop words
        stop_words = {'bir', 'bu', 'şu', 've', 'ile', 'için', 'olan', 'the', 'and', 'of', 'to', 'in'}
        
        keywords = []
        for word in words:
            if len(word) > 3 and word not in stop_words and word.isalpha():
                keywords.append(word)
        
        return keywords[:5]  # Return top 5 keywords
    
    def _analyze_code_block(self, code_content: str, language: Optional[str]) -> CodeContext:
        """Analyze code block and determine semantic context."""
        
        # Determine code type
        code_type = "example"  # default
        if any(keyword in code_content.lower() for keyword in ['function', 'def ', 'class ', 'struct']):
            code_type = "function"
        elif any(keyword in code_content.lower() for keyword in ['config', 'settings', 'properties']):
            code_type = "config"
        elif any(keyword in code_content.lower() for keyword in ['import', 'require', 'include']):
            code_type = "import"
        
        # Determine complexity
        line_count = len(code_content.split('\n'))
        if line_count <= 5:
            complexity = "simple"
        elif line_count <= 20:
            complexity = "intermediate"
        else:
            complexity = "complex"
        
        return CodeContext(
            language=language,
            code_type=code_type,
            complexity_level=complexity,
            description=f"{language or 'code'} {code_type} ({complexity})"
        )
    
    def _determine_table_type(self, headers: List[str], rows: List[List[str]]) -> str:
        """Determine table semantic type."""
        if not headers:
            return "data"
        
        header_text = ' '.join(headers).lower()
        
        if any(word in header_text for word in ['karşılaştırma', 'comparison', 'vs', 'versus']):
            return "comparison"
        elif any(word in header_text for word in ['özet', 'summary', 'toplam', 'total']):
            return "summary"
        elif any(word in header_text for word in ['zaman', 'time', 'tarih', 'date']):
            return "timeline"
        elif len(headers) == 2:
            return "key_value"
        else:
            return "data"
    
    def _generate_table_summary(self, headers: List[str], rows: List[List[str]]) -> str:
        """Generate semantic summary of table content."""
        if not headers and not rows:
            return "Empty table"
        
        row_count = len(rows)
        col_count = len(headers) if headers else (len(rows[0]) if rows else 0)
        
        summary = f"Table with {row_count} rows and {col_count} columns"
        
        if headers:
            summary += f" showing data about {', '.join(headers[:3])}"
            if len(headers) > 3:
                summary += f" and {len(headers) - 3} more columns"
        
        return summary
    
    def _classify_list_type(self, list_node: MarkdownElement) -> str:
        """Classify list semantic type."""
        if not list_node.children:
            return "empty"
        
        item_contents = [item.content.lower() for item in list_node.children]
        all_content = ' '.join(item_contents)
        
        if any(word in all_content for word in ['adım', 'step', 'önce', 'sonra', 'first', 'then']):
            return "steps"
        elif any(word in all_content for word in ['gereksinim', 'requirement', 'özellik', 'feature']):
            return "requirements"
        elif any(word in all_content for word in ['avantaj', 'advantage', 'benefit', 'fayda']):
            return "benefits"
        elif len(list_node.children) <= 3:
            return "simple"
        else:
            return "detailed"
    
    def _classify_paragraph_type(self, content: str) -> str:
        """Classify paragraph semantic type."""
        content_lower = content.lower()
        
        if content.endswith('?'):
            return "question"
        elif any(word in content_lower for word in ['örneğin', 'example', 'mesela', 'gibi']):
            return "example"
        elif any(word in content_lower for word in ['sonuç', 'conclusion', 'özet', 'summary']):
            return "conclusion"
        elif any(word in content_lower for word in ['önemli', 'important', 'dikkat', 'warning']):
            return "important"
        elif len(content.split()) < 20:
            return "short"
        else:
            return "general"
    
    def _format_table_content(self, headers: List[str], rows: List[List[str]]) -> str:
        """Format table as readable text."""
        if not headers and not rows:
            return ""
        
        lines = []
        
        if headers:
            lines.append(' | '.join(headers))
            lines.append(' | '.join(['---'] * len(headers)))
        
        for row in rows:
            lines.append(' | '.join(row))
        
        return '\n'.join(lines)
    
    def _fallback_text_parsing(self, text: str) -> List[MarkdownElement]:
        """Simple fallback parsing when all else fails."""
        paragraphs = text.split('\n\n')
        nodes = []
        
        for para in paragraphs:
            if para.strip():
                # Simple header detection
                if para.startswith('#'):
                    level = 0
                    while level < len(para) and para[level] == '#':
                        level += 1
                    
                    content = para[level:].strip()
                    nodes.append(MarkdownElement(
                        element_type=MarkdownElementType.HEADER,
                        content=content,
                        level=min(level, 6)
                    ))
                else:
                    nodes.append(MarkdownElement(
                        element_type=MarkdownElementType.PARAGRAPH,
                        content=para.strip()
                    ))
        
        return nodes
    
    def create_semantic_sections(self, ast_nodes: List[MarkdownElement]) -> List[MarkdownSection]:
        """
        Create semantic sections from AST for intelligent chunking.
        
        Args:
            ast_nodes: List of MarkdownElement nodes
            
        Returns:
            List of MarkdownSection objects for chunking
        """
        sections = []
        
        for node in ast_nodes:
            if node.element_type == MarkdownElementType.HEADER:
                section = self._create_section_from_header(node)
                sections.append(section)
            else:
                # Top-level content without header
                section = MarkdownSection(
                    title="Content",
                    level=0,
                    content=node.content,
                    elements=[node],
                    start_line=node.start_line or 0,
                    end_line=node.end_line or 0,
                    subsections=[],
                    context={
                        'type': 'content',
                        'element_type': node.element_type.value
                    }
                )
                sections.append(section)
        
        return sections
    
    def _create_section_from_header(self, header_node: MarkdownElement) -> MarkdownSection:
        """Create section from header node and its children."""
        
        # Collect all content under this header
        content_parts = []
        child_elements = []
        subsections = []
        
        for child in header_node.children:
            if child.element_type == MarkdownElementType.HEADER:
                # Sub-header creates subsection
                subsection = self._create_section_from_header(child)
                subsections.append(subsection)
            else:
                # Content element
                content_parts.append(child.content)
                child_elements.append(child)
        
        # Combine content
        section_content = header_node.content
        if content_parts:
            section_content += '\n\n' + '\n\n'.join(content_parts)
        
        # Determine line range
        start_line = header_node.start_line or 0
        end_line = header_node.end_line or 0
        
        if child_elements:
            max_end_line = max(child.end_line or 0 for child in child_elements)
            end_line = max(end_line, max_end_line)
        
        return MarkdownSection(
            title=header_node.content,
            level=header_node.level or 1,
            content=section_content,
            elements=[header_node] + child_elements,
            start_line=start_line,
            end_line=end_line,
            subsections=subsections,
            context={
                'header_type': header_node.metadata.get('semantic_type', 'general'),
                'keywords': header_node.metadata.get('keywords', []),
                'element_count': len(child_elements),
                'has_subsections': len(subsections) > 0
            }
        )