#!/usr/bin/env python
# -*- coding:utf-8 -*-

_str_synsets_up_to_third_level = """
entity.n.01
physical_entity.n.01
abstraction.n.06
thing.n.12
object.n.01
causal_agent.n.01
matter.n.03
psychological_feature.n.01
attribute.n.02
process.n.06
group.n.01
relation.n.01
communication.n.02
measure.n.02
change.n.06
freshener.n.01
horror.n.02
jimdandy.n.02
pacifier.n.02
security_blanket.n.01
stinker.n.02
thing.n.08
whacker.n.01
otherworld.n.01
set.n.02
substance.n.04
"""

_str_synsets_up_to_second_level = """
entity.n.01
physical_entity.n.01
abstraction.n.06
thing.n.08
"""

SYNSETS_DEPTH_UP_TO_THIRD_LEVEL = list(filter(bool, map(lambda s: s.strip(), _str_synsets_up_to_third_level.split("\n"))))
SYNSETS_DEPTH_UP_TO_SECOND_LEVEL = list(filter(bool, map(lambda s: s.strip(), _str_synsets_up_to_second_level.split("\n"))))