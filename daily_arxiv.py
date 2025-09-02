#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os
import shutil
import time
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

class AgentArxivDaily:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.archive_dir = "archives"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; arXiv-Agent-Bot/1.0; Educational Use)'
        })
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'keywords': [
                    'agent', 'multi-agent', 'autonomous agent', 'intelligent agent',
                    'agent-based', 'LLM agent', 'AI agent', 'cognitive agent',
                    'software agent', 'conversational agent', 'virtual agent',
                    'reinforcement learning agent', 'planning agent', 'reasoning agent'
                ],
                'categories': ['cs.AI', 'cs.MA', 'cs.CL', 'cs.LG', 'cs.RO', 'cs.HC'],
                'request_delay': 1
            }
    
    def scrape_category_new_papers(self, category: str) -> List[Dict]:
        """Scrape new papers from a single category with pagination"""
        all_papers = []
        skip = 0
        show = 25  # Papers per page
        page_num = 1
        max_pages = 10  # Limit to prevent infinite loops
        
        print(f"📥 Scraping new papers from {category}...")
        
        while page_num <= max_pages:
            # Use the 'new' URL with pagination for daily new papers
            url = f"https://arxiv.org/list/{category}/new?skip={skip}&show={show}"
            
            try:
                print(f"  📄 Page {page_num}: {url}")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all dl sections but skip 'Replacement submissions'
                article_sections = soup.find_all('dl')
                page_papers = []
                
                # Check if we have any articles on this page
                if not article_sections:
                    print(f"  📄 Page {page_num}: No article sections found")
                    break
                
                for section in article_sections:
                    # Check if this section is about replacements and skip it
                    prev_h3 = section.find_previous_sibling('h3')
                    if prev_h3 and 'replacement' in prev_h3.get_text().lower():
                        print(f"  📄 Page {page_num}: Skipping replacement submissions section")
                        continue
                    
                    dd_elements = section.find_all('dd')
                    
                    for dd in dd_elements:
                        paper_data = self._extract_paper_metadata(dd, category)
                        if paper_data:
                            page_papers.append(paper_data)
                
                if not page_papers:
                    print(f"  📄 Page {page_num}: No papers found, stopping")
                    break
                
                print(f"  📄 Page {page_num}: Found {len(page_papers)} papers")
                all_papers.extend(page_papers)
                
                # If we got fewer papers than expected, this is likely the last page
                if len(page_papers) < show:
                    print(f"  📄 Page {page_num}: Last page detected ({len(page_papers)} < {show})")
                    break
                
                # Move to next page
                skip += show
                page_num += 1
                
                # Be polite with requests
                time.sleep(self.config['request_delay'])
                    
            except Exception as e:
                print(f"❌ Error scraping {category} page {page_num}: {e}")
                break
        
        print(f"✅ {category}: Total {len(all_papers)} papers from {page_num-1} pages")
        return all_papers
    
    def _extract_paper_metadata(self, dd_element, category: str) -> Dict:
        """Extract metadata from a paper's dd element"""
        try:
            # Get title
            title_div = dd_element.find('div', class_='list-title')
            if not title_div:
                return None
            title = title_div.get_text().replace('Title:', '').strip()
            
            # Get authors
            authors_div = dd_element.find('div', class_='list-authors')
            authors = []
            if authors_div:
                author_links = authors_div.find_all('a')
                authors = [link.get_text().strip() for link in author_links]
            
            # Get arXiv ID from previous dt element
            dt_element = dd_element.find_previous_sibling('dt')
            if not dt_element:
                return None
            
            id_link = dt_element.find('a', href=lambda x: x and '/abs/' in x if x else False)
            if not id_link:
                return None
            
            arxiv_id = id_link.get('href').split('/abs/')[-1]
            paper_url = f"https://arxiv.org/abs/{arxiv_id}"
            
            return {
                'arxiv_id': arxiv_id,
                'title': title,
                'authors': authors,
                'category': category,
                'url': paper_url,
                'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}",
                'published': datetime.now().date(),
                'abstract': None  # Will be fetched separately
            }
            
        except Exception as e:
            print(f"⚠️  Error extracting paper metadata: {e}")
            return None
    
    def fetch_paper_abstract(self, paper: Dict) -> str:
        """Fetch the complete abstract for a paper"""
        try:
            response = self.session.get(paper['url'], timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find abstract in blockquote.abstract
            abstract_block = soup.find('blockquote', class_='abstract')
            if abstract_block:
                abstract_text = abstract_block.get_text()
                # Remove "Abstract:" prefix and clean up
                abstract = abstract_text.replace('Abstract:', '').strip()
                return abstract
            
            return ""
            
        except Exception as e:
            print(f"⚠️  Error fetching abstract for {paper['arxiv_id']}: {e}")
            return ""
    
    def fetch_papers(self) -> List[Dict]:
        """Fetch papers from arXiv using web scraping with deduplication"""
        all_papers = []
        seen_arxiv_ids = set()
        
        for i, category in enumerate(self.config['categories']):
            papers = self.scrape_category_new_papers(category)
            
            # Deduplicate cross-listed papers
            unique_papers = []
            for paper in papers:
                if paper['arxiv_id'] not in seen_arxiv_ids:
                    seen_arxiv_ids.add(paper['arxiv_id'])
                    unique_papers.append(paper)
                else:
                    print(f"  🔄 Skipping duplicate cross-listed paper: {paper['arxiv_id']}")
            
            print(f"  ✅ {category}: {len(unique_papers)} unique papers ({len(papers) - len(unique_papers)} duplicates removed)")
            all_papers.extend(unique_papers)
            
            # Be polite with requests
            if i < len(self.config['categories']) - 1:
                print(f"⏳ Waiting {self.config['request_delay']} seconds...")
                time.sleep(self.config['request_delay'])
        
        # Filter for agent-related papers
        agent_papers = []
        for paper in all_papers:
            if self.is_agent_relevant(paper):
                agent_papers.append(paper)
        
        print(f"📊 Total unique papers scraped: {len(all_papers)}")
        print(f"🤖 Agent-related papers: {len(agent_papers)}")
        
        # Fetch abstracts for agent papers
        if agent_papers:
            agent_papers = self.fetch_abstracts_batch(agent_papers)
        
        return agent_papers
    
    def fetch_abstracts_batch(self, papers: List[Dict], max_workers: int = 3) -> List[Dict]:
        """Fetch abstracts for multiple papers using thread pool"""
        print(f"📝 Fetching abstracts for {len(papers)} papers...")
        
        def fetch_with_delay(paper):
            time.sleep(self.config['request_delay'])  # Rate limiting
            abstract = self.fetch_paper_abstract(paper)
            paper['abstract'] = abstract
            return paper
        
        completed_papers = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paper = {executor.submit(fetch_with_delay, paper): paper for paper in papers}
            
            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    completed_paper = future.result()
                    completed_papers.append(completed_paper)
                    if len(completed_papers) % 5 == 0:
                        print(f"📝 Fetched {len(completed_papers)}/{len(papers)} abstracts...")
                except Exception as e:
                    print(f"❌ Error fetching abstract for {paper['arxiv_id']}: {e}")
                    paper['abstract'] = ""
                    completed_papers.append(paper)
        
        return completed_papers
    
    def is_agent_relevant(self, paper: Dict) -> bool:
        """Check if paper is relevant to agent research with improved matching"""
        title = paper['title'].lower()
        abstract = (paper.get('abstract') or '').lower()
        
        # Enhanced keywords list
        enhanced_keywords = self.config['keywords'] + [
            'agentic', 'agentive', 'multi agent', 'multiagent',
            'autonomous system', 'intelligent system', 'embodied',
            'rl agent', 'agents', 'cooperative', 'collaborative agent',
            'virtual agent', 'chatbot', 'dialogue system'
        ]
        
        # Check keywords in title (higher priority)
        for keyword in enhanced_keywords:
            if keyword.lower() in title:
                return True
        
        # Check keywords in abstract (if no title match)
        for keyword in enhanced_keywords:
            if keyword.lower() in abstract:
                return True
        
        # Additional context-based checks
        agent_contexts = [
            'agent-based', 'multi-agent', 'agent learning', 'agent planning',
            'agent reasoning', 'agent behavior', 'agent interaction',
            'reinforcement learning', 'autonomous', 'intelligent'
        ]
        
        for context in agent_contexts:
            if context in title or context in abstract:
                return True
        
        return False
    
    def categorize_paper(self, paper: Dict) -> str:
        """Categorize paper based on title and abstract"""
        title_abstract = (paper['title'] + " " + (paper.get('abstract') or '')).lower()
        
        categories = {
            'Multi-Agent Systems': ['multi-agent', 'multi agent', 'cooperative agent', 'agent cooperation', 'collaborative agent'],
            'LLM Agents': ['llm agent', 'language model agent', 'chatgpt agent', 'gpt agent', 'large language model agent'],
            'Agent Applications': [
                'autonomous agent', 'autonomous system', 'self-driving',
                'conversational agent', 'chatbot', 'dialogue agent', 'virtual assistant',
                'game playing', 'game agent'
            ],
            'Reinforcement Learning': [
                'reinforcement learning', 'deep reinforcement learning', 'rl agent', 'policy optimization',
                'q-learning', 'actor-critic', 'policy gradient', 'temporal difference',
                'markov decision process', 'mdp', 'proximal policy optimization', 'ppo',
                'deep q-network', 'dqn', 'soft actor-critic', 'sac'
            ],
            'Benchmarks and Datasets': [
                'benchmark', 'dataset', 'evaluation', 'testbed', 'suite',
                'benchmark for', 'dataset for', 'evaluation of', 'test suite',
                'benchmarking', 'evaluation framework', 'evaluation benchmark'
            ],
            'Planning and Reasoning': ['agent planning', 'reasoning agent', 'decision making', 'strategic reasoning'],
            'Other Agent Research': []
        }
        
        for category, keywords in categories.items():
            if category == 'Other Agent Research':
                continue
            for keyword in keywords:
                if keyword in title_abstract:
                    return category
        
        return 'Other Agent Research'
    
    def extract_repo_url(self, abstract: str) -> str:
        """Extract repository URL from abstract using regex"""
        import re
        
        # Common patterns for repository URLs
        patterns = [
            r'https://github\.com/[\w\-\.]+/[\w\-\.]+',
            r'github\.com/[\w\-\.]+/[\w\-\.]+',
            r'https://gitlab\.com/[\w\-\.]+/[\w\-\.]+',
            r'https://bitbucket\.org/[\w\-\.]+/[\w\-\.]+',
            r'https://code\.google\.com/[\w\-\.]+',
            r'https://huggingface\.co/[\w\-\.]+/[\w\-\.]+'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, abstract, re.IGNORECASE)
            if matches:
                # Return the first match, ensure it has https://
                url = matches[0]
                if not url.startswith('https://'):
                    url = 'https://' + url
                return url
        
        return ""
    
    def format_paper(self, paper: Dict) -> str:
        """Format paper information for markdown output with collapsible format"""
        authors = paper.get('authors', [])
        if authors:
            author_str = ', '.join(authors)  # Show all authors
        else:
            author_str = 'Unknown Authors'
        
        # Extract repository URL from abstract
        abstract = paper.get('abstract', '')
        repo_url = self.extract_repo_url(abstract)
        
        # Build links section
        links = []
        if repo_url:
            links.append(f"[[repo]]({repo_url})")
        links.append(f"[[pdf]]({paper['pdf_url']})")
        links_str = " ".join(links)
        
        return f"""<details>
<summary><strong>{paper['title']}</strong> - {author_str} - {links_str}</summary>

**Abstract:** {abstract}

**arXiv ID:** {paper['arxiv_id']}
</details>

"""
    
    def generate_markdown(self, papers: List[Dict]) -> str:
        """Generate markdown content for README"""
        # Group papers by category
        categorized_papers = {}
        for paper in papers:
            category = self.categorize_paper(paper)
            if category not in categorized_papers:
                categorized_papers[category] = []
            categorized_papers[category].append(paper)
        
        markdown = f"# Agent Research Papers Daily\n\n"
        markdown += f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown += f"**Total Papers:** {len(papers)}\n\n"
        
        # Table of Contents
        markdown += "## Table of Contents\n\n"
        for category in sorted(categorized_papers.keys()):
            markdown += f"- [{category}](#{category.lower().replace(' ', '-').replace('-', '-')})\n"
        markdown += "\n"
        
        # Papers by category with collapsible sections
        for category in sorted(categorized_papers.keys()):
            papers_in_category = categorized_papers[category]
            
            markdown += f"""<details open>
<summary><h2>{category} ({len(papers_in_category)} papers)</h2></summary>

"""
            
            for paper in papers_in_category:
                markdown += self.format_paper(paper)
            
            markdown += "</details>\n\n"
        
        # Footer
        markdown += "---\n\n"
        markdown += "*This list is automatically generated daily using arXiv API*\n"
        
        return markdown
    
    def archive_previous_readme(self):
        """Archive the current README.md if it exists"""
        if not os.path.exists("README.md"):
            return
        
        # Create archive directory if it doesn't exist
        os.makedirs(self.archive_dir, exist_ok=True)
        
        # Generate archive filename with current date
        today = datetime.now().strftime('%Y-%m-%d')
        archive_filename = f"README-{today}.md"
        archive_path = os.path.join(self.archive_dir, archive_filename)
        
        # Copy current README to archive
        shutil.copy2("README.md", archive_path)
        print(f"Archived previous README to {archive_path}")
    
    def update_readme(self, content: str, output_file: str = "README.md"):
        """Update README file with new content"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def run(self):
        """Main execution method"""
        print("Archiving previous README.md...")
        self.archive_previous_readme()
        
        print("Fetching agent-related papers from arXiv...")
        papers = self.fetch_papers()
        print(f"Found {len(papers)} papers")
        
        print("Generating markdown content...")
        markdown_content = self.generate_markdown(papers)
        
        print("Updating README.md...")
        self.update_readme(markdown_content)
        
        print("Done!")

if __name__ == "__main__":
    agent_arxiv = AgentArxivDaily()
    agent_arxiv.run()