from .agent import PGAgent


def get_parser_agent_by_name(name):
    if name == 'vanilla':
        return PGAgent
    else:
        raise ValueError(name)
