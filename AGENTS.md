0. Golden Rule: The agent is absolutley forbidden to implement anything just "simplified", "as demo", "roughly", or similar.
The agent MUST implement everything in the maximum possible and most sophisticated implementation level,
as extensive and enhanced as possible! ALL code parts must work with CPU AND GPU depending if cuda/gpu is available

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
