"""GitHub Learner — Skill for autonomous learning from code repositories.

Allows Enton to search for repositories, read READMEs, and learn new technical concepts
when bored or curious.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from agno.tools import Toolkit

logger = logging.getLogger(__name__)

@dataclass
class RepoInfo:
    name: str
    url: str
    description: str

class GitHubLearner(Toolkit):
    """Tools for finding and learning from GitHub repositories."""

    def __init__(self):
        super().__init__(name="github_learner")
        self.register(self.study_github_topic)

    def study_github_topic(self, topic: str) -> str:
        """
        Pesquisa e estuda um tópico técnico no GitHub.
        
        Usa busca pública para encontrar repositórios relevantes, lê seus READMEs
        e retorna um resumo do que foi aprendido.
        
        Args:
            topic: O tema técnico para estudar (ex: "rust actix", "transformer architecture").
            
        Returns:
            Um resumo do que foi aprendido com base nos repositórios encontrados.
        """
        logger.info("Studying topic on GitHub: %s", topic)
        
        # 1. Search for repos
        repos = self._search_repos(topic)
        if not repos:
            return f"Não encontrei repositórios interessantes sobre '{topic}' no momento."

        results = []
        # 2. Read top 2 repos
        for repo in repos[:2]:
            content = self._read_readme(repo.url)
            if content:
                # Truncate content to avoid token overflow
                snippet = content[:2000] + "\n...(truncated)..." if len(content) > 2000 else content
                results.append(f"## Repository: {repo.name}\n{repo.description}\n\nRunning Notes:\n{snippet}")
        
        if not results:
            return f"Encontrei repos sobre '{topic}' mas não consegui ler os READMEs."
            
        return f"# Estudo sobre: {topic}\n\n" + "\n\n".join(results)

    def _search_repos(self, query: str) -> list[RepoInfo]:
        """Search GitHub via public API."""
        try:
            import httpx
            # Using GitHub Search API (public limit: 10/min, usually enough for slow browsing)
            url = "https://api.github.com/search/repositories"
            resp = httpx.get(
                url, 
                params={"q": query, "sort": "stars", "order": "desc", "per_page": 3},
                headers={"User-Agent": "Enton-AI-Agent"},
                timeout=10.0
            )
            
            if resp.status_code != 200:
                logger.warning("GitHub API error: %s", resp.status_code)
                return []
                
            data = resp.json()
            items = data.get("items", [])
            
            return [
                RepoInfo(
                    name=item["full_name"],
                    url=item["html_url"],
                    description=item.get("description") or "No description"
                )
                for item in items
            ]
        except Exception as e:
            logger.error("Failed to search GitHub: %s", e)
            return []

    def _read_readme(self, repo_url: str) -> str | None:
        """Fetch raw README content."""
        # Convert https://github.com/user/repo -> https://raw.githubusercontent.com/user/repo/HEAD/README.md
        # Basic heuristic, might fail for non-main branches or different filenames
        try:
            import httpx
            
            # Extract user/repo
            match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
            if not match:
                return None
                
            user, repo = match.groups()
            
            # Try common branches and filenames
            variants = [
                f"https://raw.githubusercontent.com/{user}/{repo}/HEAD/README.md",
                f"https://raw.githubusercontent.com/{user}/{repo}/master/README.md",
                f"https://raw.githubusercontent.com/{user}/{repo}/main/README.md",
            ]
            
            for url in variants:
                resp = httpx.get(url, timeout=5.0)
                if resp.status_code == 200:
                    return resp.text
                    
            return None
        except Exception as e:
            logger.error("Failed to read README for %s: %s", repo_url, e)
            return None
