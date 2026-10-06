{% for category, _ in definitions.items() if category in sections[''] %}

## {{ definitions[category]['name'] }}
{% for text, pull_requests in sections[''][category].items() | reverse %}
- {{ text }}{{ (" " ~ pull_requests | join(" ")) if pull_requests else "" }}
{% endfor %}
{% endfor %}
{{ "\n" }}
