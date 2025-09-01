#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arxiv
import yaml
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import re

class AgentArxivDaily:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.client = arxiv.Client()
        
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
                    'software agent', 'conversational agent', 'virtual agent'
                ],
                'categories': ['cs.AI', 'cs.MA', 'cs.CL', 'cs.LG'],
                'max_results': 50,
                'days_back': 7
            }
    
    def build_query(self) -> str:
        """Build arXiv search query based on keywords and categories"""
        keyword_query = ' OR '.join([f'"{keyword}"' for keyword in self.config['keywords']])
        category_query = ' OR '.join([f'cat:{cat}' for cat in self.config['categories']])
        
        return f"({keyword_query}) AND ({category_query})"
    
    def fetch_papers(self) -> List[arxiv.Result]:
        """Fetch papers from arXiv based on configuration"""
        query = self.build_query()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['days_back'])
        
        search = arxiv.Search(
            query=query,
            max_results=self.config['max_results'],
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for result in self.client.results(search):
            # Filter by date
            if result.published.date() >= start_date.date():
                papers.append(result)
        
        return sorted(papers, key=lambda x: x.published, reverse=True)
    
    def categorize_paper(self, paper: arxiv.Result) -> str:
        """Categorize paper based on title and abstract"""
        title_abstract = (paper.title + " " + paper.summary).lower()
        
        categories = {
            'Multi-Agent Systems': ['multi-agent', 'multi agent', 'cooperative agent', 'agent cooperation'],
            'LLM Agents': ['llm agent', 'language model agent', 'chatgpt agent', 'gpt agent'],
            'Autonomous Agents': ['autonomous agent', 'autonomous system', 'self-driving'],
            'Conversational Agents': ['conversational agent', 'chatbot', 'dialogue agent', 'virtual assistant'],
            'Game Playing Agents': ['game playing', 'game agent', 'reinforcement learning agent'],
            'Planning and Reasoning': ['agent planning', 'reasoning agent', 'decision making'],
            'Agent Learning': ['agent learning', 'learning agent', 'adaptive agent'],
            'Other Agent Research': []
        }
        
        for category, keywords in categories.items():
            if category == 'Other Agent Research':
                continue
            for keyword in keywords:
                if keyword in title_abstract:
                    return category
        
        return 'Other Agent Research'
    
    def format_paper(self, paper: arxiv.Result, category: str) -> str:
        """Format paper information for markdown output"""
        authors = ', '.join([author.name for author in paper.authors[:3]])
        if len(paper.authors) > 3:
            authors += ' et al.'
        
        return f"- **{paper.title}** - {authors} - [{paper.published.strftime('%Y-%m-%d')}]({paper.pdf_url})\n"
    
    def generate_markdown(self, papers: List[arxiv.Result]) -> str:
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
        
        # Papers by category
        for category in sorted(categorized_papers.keys()):
            papers_in_category = categorized_papers[category]
            markdown += f"## {category}\n\n"
            markdown += f"*{len(papers_in_category)} papers*\n\n"
            
            for paper in papers_in_category:
                markdown += self.format_paper(paper, category)
            
            markdown += "\n"
        
        # Footer
        markdown += "---\n\n"
        markdown += "*This list is automatically generated daily using arXiv API*\n"
        
        return markdown
    
    def update_readme(self, content: str, output_file: str = "README.md"):
        """Update README file with new content"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def run(self):
        """Main execution method"""
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