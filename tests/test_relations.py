import time
import unittest
from unittest.mock import patch
from voxhands.planner import build_plan
from voxhands.safety import validate_plan
from voxhands.simulation import TableSettingSimulation

class RelationTests(unittest.TestCase):
    def test_aliases_and_fit(self):
        for text in ('put cup onto plate', 'put the mug on the blue plate'):
            plan = build_plan(text)
            self.assertEqual(validate_plan(plan), [], text)
            self.assertEqual(len(plan.actions), 1)
            self.assertEqual(plan.actions[0].target_id, 'plate_surface')
        for text in ('put plate onto plate', 'put plate into cup', 'put cup onto spoon', 'place fork on plate', 'put spoon onto plate'):
            self.assertTrue(validate_plan(build_plan(text)), text)

    def test_place_remove_and_support_guard(self):
        with patch('voxhands.simulation.RUN_DURATION_MS', 150):
            sim = TableSettingSimulation()
            def run(text):
                state = sim.submit_command(text)
                self.assertEqual(state['status'], 'running', state.get('plan'))
                deadline = time.monotonic() + 3
                while sim.snapshot()['status'] == 'running' and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(sim.snapshot()['status'], 'complete')
            run('put cup onto plate')
            state = sim.snapshot()
            self.assertEqual(state['objects']['cup']['supported_by'], 'blue_plate')
            self.assertEqual(state['objects']['cup']['x'], state['objects']['blue_plate']['x'])
            self.assertEqual(sim.submit_command('move plate left')['status'], 'blocked')
            run('move cup right')
            self.assertNotIn('supported_by', sim.snapshot()['objects']['cup'])
            run('put spoon into mug')
            self.assertEqual(sim.snapshot()['objects']['spoon']['container'], 'cup')

    def test_cup_on_plate_then_spoon_inside(self):
        with patch('voxhands.simulation.RUN_DURATION_MS', 150):
            sim = TableSettingSimulation()
            for command in ('put cup onto plate', 'put spoon into cup'):
                state = sim.submit_command(command)
                self.assertEqual(state['status'], 'running', state['plan']['safety_issues'])
                deadline = time.monotonic() + 3
                while sim.snapshot()['status'] == 'running' and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(sim.snapshot()['status'], 'complete')
            state = sim.snapshot()
            self.assertEqual(state['objects']['cup']['supported_by'], 'blue_plate')
            self.assertEqual(state['objects']['spoon']['container'], 'cup')
            self.assertEqual(sim.submit_command('move cup right')['status'], 'blocked')

    def test_unrelated_overlap_still_blocks_insertion(self):
        sim = TableSettingSimulation()
        sim.objects['fork'].update(x=sim.objects['cup']['x'], y=sim.objects['cup']['y'])
        state = sim.submit_command('put spoon into cup')
        self.assertEqual(state['status'], 'blocked')
        self.assertIn('Fork', ' '.join(state['plan']['safety_issues']))
