"""Conversation workflow route names.

The runtime has two normal response paths only: realtime queries perform web
search before generation, while every other query goes directly to the common
LLM generator.
"""

ROUTE_BRANCH_REALTIME = "realtime"
ROUTE_BRANCH_GENERAL = "general"
