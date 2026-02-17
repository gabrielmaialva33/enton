import pytest
from unittest.mock import MagicMock, patch
from enton.skills.github_learner import GitHubLearner

def test_learner_initialization():
    learner = GitHubLearner()
    assert learner.name == "github_learner"

@patch("httpx.get")
def test_search_repos_success(mock_get):
    learner = GitHubLearner()
    
    # Mock response for search
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "items": [
            {
                "full_name": "user/repo1",
                "html_url": "https://github.com/user/repo1",
                "description": "Test Repo 1"
            }
        ]
    }
    mock_get.return_value = mock_resp
    
    repos = learner._search_repos("test")
    assert len(repos) == 1
    assert repos[0].name == "user/repo1"

@patch("httpx.get")
def test_read_readme_success(mock_get):
    learner = GitHubLearner()
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "# README Content"
    mock_get.return_value = mock_resp
    
    content = learner._read_readme("https://github.com/user/repo1")
    assert content == "# README Content"

@patch("httpx.get")
def test_study_topic_integration(mock_get):
    """Test full flow with mocks."""
    learner = GitHubLearner()
    
    # Setup mocks for search AND read
    # This is tricky because search and read use httpx.get with different URLs.
    # We can use side_effect based on URL arg or call count.
    
    def side_effect(*args, **kwargs):
        url = args[0]
        resp = MagicMock()
        if "api.github.com/search" in url:
            resp.status_code = 200
            resp.json.return_value = {
                "items": [
                    {"full_name": "u/r", "html_url": "https://github.com/u/r", "description": "d"}
                ]
            }
        elif "raw.githubusercontent.com" in url:
            resp.status_code = 200
            resp.text = "# Learned Content"
        else:
            resp.status_code = 404
        return resp
        
    mock_get.side_effect = side_effect
    
    summary = learner.study_github_topic("test topic")
    assert "Learned Content" in summary
    assert "Estudo sobre: test topic" in summary
