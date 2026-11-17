#!/usr/bin/env python3
"""Independent synthetic RL-task verifier, inspired by Steadyworks' internship.

No affiliation, real tickets, model calls, network calls or state mutation.
Demo: python verifier.py | Tests: python verifier.py --test
JSON file: python verifier.py input.json
Input: {before:{ticket:state},after:{ticket:state},goals:{ticket:patch}}
A state is a JSON object; a patch is a nonempty subset of desired fields.
All fields outside the goal patch must be preserved, and unscoped tickets
must be unchanged. Exact types matter (True is not the integer 1).
Reward1 requires a valid state transition, not an agent's success message.
This deliberately narrow contract is not a general task-success oracle.
Production needs trusted snapshots, schemas, authorization and domain policy.
"""
import copy
import json
import sys


def equal(a, b):
    if type(a) is not type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def verify(payload):
    reasons = []
    if not isinstance(payload, dict): return {'reward': 0, 'reasons': ['invalid payload']}
    before, after, goals = (payload.get(k) for k in ('before', 'after', 'goals'))
    if any(not isinstance(x, dict) for x in (before, after, goals)) or not goals:
        return {'reward': 0, 'reasons': ['expected before/after objects and nonempty goals']}
    if any(not isinstance(k, str) or not isinstance(v, dict) for m in (before, after) for k, v in m.items()):
        return {'reward': 0, 'reasons': ['invalid ticket state']}
    if before.keys() != after.keys(): reasons.append('ticket set changed')
    for ticket in sorted(before):
        if ticket not in after: continue
        if ticket not in goals:
            if not equal(before[ticket], after[ticket]): reasons.append('unscoped ticket changed: ' + ticket)
            continue
        patch = goals[ticket]
        if not isinstance(patch, dict) or not patch:
            reasons.append('invalid goal patch: ' + ticket)
            continue
        if not set(patch).issubset(before[ticket]):
            reasons.append('unknown goal fields: ' + ticket)
            continue
        expected = copy.deepcopy(before[ticket]); expected.update(patch)
        if not equal(after[ticket], expected): reasons.append('goal not met or extra fields changed: ' + ticket)
        if equal(before[ticket], expected): reasons.append('goal already satisfied at start: ' + ticket)
    for ticket in sorted(set(goals) - set(before)): reasons.append('unknown goal ticket: ' + ticket)
    return {'reward': int(not reasons), 'reasons': reasons,
            'contract': 'Exact goal patch; preserve all out-of-scope state.'}


def fixture():
    before = {'T1': {'status': 'open', 'owner': 'queue-A', 'version': 1},
              'T2': {'status': 'open', 'owner': 'queue-B', 'version': 1}}
    after = copy.deepcopy(before); after['T1']['status'] = 'resolved'
    return {'before': before, 'after': after, 'goals': {'T1': {'status': 'resolved'}}}


def tests():
    import unittest
    class Verifier(unittest.TestCase):
        def test_success(self): self.assertEqual(verify(fixture())['reward'], 1)
        def test_no_mutation(self):
            p=fixture(); old=copy.deepcopy(p); verify(p); self.assertEqual(old,p)
        def test_claim_only(self):
            p=fixture(); p['after']=copy.deepcopy(p['before']); p['agent_says']='Done'; self.assertEqual(verify(p)['reward'],0)
        def test_scope(self):
            p=fixture(); p['after']['T2']['status']='resolved'; self.assertEqual(verify(p)['reward'],0)
        def test_extra_field(self):
            p=fixture(); p['after']['T1']['proof']='looks good'; self.assertEqual(verify(p)['reward'],0)
        def test_deleted_ticket(self):
            p=fixture(); del p['after']['T2']; self.assertEqual(verify(p)['reward'],0)
        def test_added_ticket(self):
            p=fixture(); p['after']['T3']={}; self.assertEqual(verify(p)['reward'],0)
        def test_type_confusion(self):
            p=fixture(); p['after']['T1']['version']=True; self.assertEqual(verify(p)['reward'],0)
        def test_already_solved(self):
            p=fixture(); p['before']=copy.deepcopy(p['after']); self.assertEqual(verify(p)['reward'],0)
        def test_unknown_goal(self):
            p=fixture(); p['goals']['T9']={'status':'resolved'}; self.assertEqual(verify(p)['reward'],0)
        def test_empty_goal(self):
            p=fixture(); p['goals']={}; self.assertEqual(verify(p)['reward'],0)
        def test_bad_root(self): self.assertEqual(verify([])['reward'],0)
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Verifier))
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--test']: tests()
    else:
        if len(sys.argv)>1:
            with open(sys.argv[1],encoding='utf-8') as stream: payload=json.load(stream)
        else: payload=fixture()
        print(json.dumps(verify(payload),indent=2))
