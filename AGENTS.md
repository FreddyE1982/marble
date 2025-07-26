0. Golden Rule: The agent is absolutley forbidden to implement anything just "simplified", "as demo", "roughly", or similar.
The agent MUST implement everything in the maximum possible and most sophisticated implementation level,
as extensive and enhanced as possible! ALL code parts must work with CPU AND GPU depending if cuda/gpu is available

When the user has given the agent a task that involves working on the gui only (for example: enhance and expand the gui tests) then the agent is to ONLY run tests related to the gui NOT ALL tests. 
In this case the rule that says to always run all tests is temporarily not in effect.

Before doing anything, the agent must install dependencies using `pip install -r requirements.txt`.
If there are conflicts with in requirements.txt the agent must fix them.
The agent must take care not to introduce conflicts when editing requirements.txt

1. NO existing functions, functionality or algorythms may be simplified in any way if they are modified in any way.
2. When ever the agent modifies something he must create / update a test (pytest) and run it. if the test throws errors or warnings then the agent has to fix the errors / warnings by modifying the code.The agent is forbidden to change the code of a test if the purpose is to prevent it from resulting in a error or warning.
3. The must be tests for all functions and algorythims...each seperately AND ALL OF IT IN CONCERT
4. The agent must update `requirements.txt` automatically after all changes.
5. Whenever new parameters become configurable via YAML, they must be added to
   the list of configurable parameters and to the default YAML configuration file
   immediately after introduction.
6. The agent must maintain a `yaml-manual.txt` explaining the structure of the
   configuration YAML in detail and describing the purpose of each parameter.
   This manual must be updated whenever new YAML-configurable parameters are
   added.
7. Descriptions in `yaml-manual.txt` must be thorough. Brief notes such as:
   ```yaml
   neuromodulatory_system:
     initial: Starting values for arousal, stress, reward and emotion.
   ```
   are insufficient. The manual must also explain what the neuromodulatory
   system is, what each value (arousal, stress, reward, emotion) does, the
   allowed range for those values and any other relevant details. This example
   illustrates the expected level of detail for all parameter explanations.
8. `yaml-manual.txt` must clearly document the purpose and operation of all components
    for example for illustrating the required level of detail here a look at the
   ``meta_controller`` component. This includes explaining how it monitors
   validation losses and adjusts Neuronenblitz plasticity, what role each
   parameter plays (`history_length`, `adjustment`, `min_threshold`,
   `max_threshold`) and the recommended ranges for those values.
9. The agent must maintain `TUTORIAL.md` containing a detailed step-by-step tutorial. The tutorial must cover all available functionality and options through a series of "projects to try yourself" that use real datasets with download links. Projects should be ordered from easiest to most advanced. Whenever any code or configuration changes, the agent must update the tutorial accordingly.
10. ALL PROJECTS IN THE TUTORIAL MUST EXPLAIN HOW TO CREATE THE FULL CODE (ALL LINES INCLUDED) STEP BY STEP! ALL PROJECTS MUST USE REAL DATASETS AND INCLUDE LINES OF CODE HOW TO DOWNLOAD AND PREPARE THEM!
11. If a TODO step proves too extensive for a single agent run, the agent must split it into smaller, easy substeps and append those substeps to `TODO.md` before continuing.

streamlit gui
ALL functionality MUST be available via the GUI. The assistant groups functionality logically in tabs. It uses the existing tabs for grouping of functionality when possible but creates new tabs if that seems more sensible. The agent must ensure that it never creates a new tab for things that can be logically put into a existing tab. for example: there may not be multiple tabs containing statistics..that would all belong into ONE tab.
The agent must ensure that the streamlit GUI will show correctly on desktop AND mobile phones. the correct "mode" to show in should be recognised automatically. you may NOT remove or simplify ANY part of the gui in ANY mode.
The agent must use st.expander to group multiple things that belong to the same functionality or workflow together in a tab as explained here: https://docs.streamlit.io/develop/api-reference/layout/st.expander
The agent must use st.dialog where appropriate as explained here: https://docs.streamlit.io/develop/api-reference/execution-flow/st.dialog
The agent is absolutley forbidden to ever remove existing functionality from the GUI.
group things where it makes sense logically together using expandables. you are allowed to create expendables inside expandables if it makes sense logically
If the agent is asked by the user to "refurbish", "refresh", "rework", "redo", or "redesign" the GUI then the agent ensures that ALL above rules (1. - 6.) about the streamlit gui are strictly adhered to but those rules ALSO apply WHENEVER the agent works on the application in any way.
The agent is to unittest the streamlit gui. How this can be done is explained in the file "streamlittestinghowto.md" found in the repo. The streamlit gui is big but that is not to discourage the agent. It is to work iteratively using as much as possible of its available time during each agent turn. The goal is that the ENTIRE streamlit gui is thoroughly tested. NO parts of the gui may be skipped during testing. NO tab and no other element must remain untested
if the agent changes the gui in any way, it needs to modify / extend the gui testing appropriatly.
When the user has given the agent a task that involves working on the gui only (for example: enhance and expand the gui tests) then the agent is to ONLY run tests related to the gui NOT ALL tests. In this case the rule that says to always run all tests is temporarily not in effect.
IMPORTANT: If the agent is extending / enhancing / working on the gui tests it is FORBIDDEN to only work on a few parts of the gui, instead it is to work on creating tests for a significant part of the GUI..for example a whole tab or at least half of a tab.
