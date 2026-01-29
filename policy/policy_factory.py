from policy.orca import ORCA
from policy.orca_hard import ORCA as ORCA_hard

policy_factory = dict()
policy_factory['orca'] = ORCA
policy_factory['orca_hard'] = ORCA_hard

