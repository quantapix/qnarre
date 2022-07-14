# Copyright 2019 Quantapix Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from traitlets.config.loader import Config
from IPython.terminal.prompts import Prompts, Token
from IPython.terminal.embed import InteractiveShellEmbed


class Prompt(Prompts):
    def in_prompt_tokens(self, cli=None):
        return (
            (Token.Prompt, 'In <'),
            (Token.PromptNum, str(self.shell.execution_count)),
            (Token.Prompt, '>: '),
        )

    def out_prompt_tokens(self):
        return (
            (Token.OutPrompt, 'Out<'),
            (Token.OutPromptNum, str(self.shell.execution_count)),
            (Token.OutPrompt, '>: '),
        )


try:
    get_ipython
except NameError:
    nested = 0
    cfg = Config()
    cfg.TerminalInteractiveShell.prompts_class = Prompt
else:
    print("Running nested copies of IPython.")
    print("The prompts for the nested copy have been modified")
    cfg = Config()
    nested = 1

ipshell = InteractiveShellEmbed(
    config=cfg, banner1='Entering IPython', exit_msg='Exiting IPython...')

ipshell2 = InteractiveShellEmbed(config=cfg, banner1='IPython again')

# ipshell('***Called from top level. '
#         'Hit Ctrl-D to exit interpreter and continue program.\n'
#        'Note that if you use %kill_embedded, you can fully deactivate\n'
#        'This embedded instance so it will never turn on again')

# usage:
# from .ipython import ipshell
# ipshell()
