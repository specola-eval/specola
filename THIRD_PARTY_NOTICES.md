# Third-party notices

Specola is released under the MIT licence in [`LICENSE`](LICENSE). Some
files in this repository contain material written by others. This file
lists that material: what it is, where it comes from, its licence and
copyright notice, and which files here carry it. The full licence texts
follow at the end, each as its upstream ships it.

A built wheel carries this file in its `specola-<version>.dist-info/`
directory, in the `licenses/` subdirectory when `setuptools` 77 or newer
built it. Packages that pip installs next to Specola carry their own
notices and are not listed here.

Every licence text and copyright line below was read on 2026-10-13 from
the upstream repository or package at the version named, at the address
each entry lists under Source.

## MuJoCo

- Versions: 3.2.3, 3.3.0, 3.3.1, 3.8.1 and 3.10.0 (git tags of the same
  names).
- Upstream: <https://github.com/google-deepmind/mujoco>
- Licence: Apache License 2.0 for source code, and Creative Commons
  Attribution 4.0 International (CC BY 4.0) for the ReStructuredText
  documents in `doc/`. The "License and Disclaimer" section of MuJoCo's
  `README.md` states this split, in identical words at all five tags.
- Copyright: `Copyright 2021 DeepMind Technologies Limited.` (README).
  The headers of the excerpted source files carry these lines:

  ```text
  Copyright 2021 DeepMind Technologies Limited
  Copyright 2022 DeepMind Technologies Limited
  Copyright 2024 DeepMind Technologies Limited
  Copyright 2025 DeepMind Technologies Limited
  Copyright 2026 DeepMind Technologies Limited
  ```

- NOTICE file: none. No tag of the five has a NOTICE file at its root, so
  Apache License 2.0 section 4(d) adds no text.
- Files here: 30 JSON files in `src/specola/data/` quote MuJoCo verbatim,
  in their `verbatim`, `raw_comment` and `meaning_verbatim` fields. Each
  excerpt is unmodified and records its upstream path and line range; the
  version is in the file name.
  - `mujoco-completion-<version>.json`, five versions: source (`src/`).
  - `mujoco-fields-<version>.json`, five versions: headers (`include/`),
    source, Python bindings (`python/`) and documentation (`doc/`).
  - `mujoco-projection-<version>.json`, five versions: headers, source and
    one documentation excerpt.
  - `mujoco-query-metadata-<version>.json`, five versions: headers, source
    and two documentation excerpts.
  - `mujoco-spatial-queries-<version>.json`, five versions: source.
  - `mujoco-native-<version>.json` for 3.2.3, 3.3.0, 3.3.1 and 3.10.0:
    Python bindings.
  - `mujoco-stack-3.10.0.json`: headers and Python bindings.
- Documentation excerpts (CC BY 4.0): MuJoCo documentation by DeepMind
  Technologies Limited, from `doc/` at the tag named in each file's
  version (for example
  <https://github.com/google-deepmind/mujoco/tree/3.10.0/doc>), licensed
  under CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/legalcode>.
  The excerpts are quoted without change.
- Source: `LICENSE` at
  `https://raw.githubusercontent.com/google-deepmind/mujoco/<tag>/LICENSE`,
  identical at all five tags (SHA-256
  `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`);
  `README.md` at each tag; the header of each of the 222 excerpted
  upstream files at its tag.
- Licence text: [Apache License 2.0](#apache-license-20).

## robosuite

- Version: commit `eafb81f54ffc104f905ee48a16bb15f059176ad3`.
- Upstream: <https://github.com/ARISE-Initiative/robosuite>
- Licence: MIT.
- Copyright: `Copyright (c) 2022 Stanford Vision and Learning Lab and UT
  Robot Perception and Learning Lab`
- Files here: `quat2axisangle` in `benchmarks/x7_oft_server.py` and in
  `benchmarks/x7_openvla_server.py`. Both copies came through the policy
  repositories below, whose `experiments/robot/libero/libero_utils.py`
  marks the function "Copied from robosuite" with a link to
  `robosuite/utils/transform_utils.py` at this commit.
- Source:
  <https://raw.githubusercontent.com/ARISE-Initiative/robosuite/eafb81f54ffc104f905ee48a16bb15f059176ad3/LICENSE>
- Licence text: [MIT, robosuite](#mit-robosuite).

## openvla-oft

- Version: commit `e4287e94541f459edc4feabc4e181f537cd569a8`.
- Upstream: <https://github.com/moojink/openvla-oft>
- Licence: MIT.
- Copyright: `Copyright (c) 2025 Moo Jin Kim, Chelsea Finn, Percy Liang.`
- Files here: `quat2axisangle` in `benchmarks/x7_oft_server.py`, from
  `experiments/robot/libero/libero_utils.py` (the copy first converts its
  argument to a new float64 array).
- Source:
  <https://raw.githubusercontent.com/moojink/openvla-oft/e4287e94541f459edc4feabc4e181f537cd569a8/LICENSE>
- Licence text: [MIT, openvla-oft](#mit-openvla-oft).

## OpenVLA

- Version: commit `c8f03f48af692657d3060c19588038c7220e9af9`.
- Upstream: <https://github.com/openvla/openvla>
- Licence: MIT.
- Copyright: `Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth
  Karamcheti.`
- Files here: the resize and quaternion helpers in
  `benchmarks/x7_openvla_server.py`, from
  `experiments/robot/libero/libero_utils.py`.
- Source:
  <https://raw.githubusercontent.com/openvla/openvla/c8f03f48af692657d3060c19588038c7220e9af9/LICENSE>
- Licence text: [MIT, OpenVLA](#mit-openvla).

## VLA-Adapter

- Version: commit `23fa0c9c159e2aa04341cdd3e924f44061311060`.
- Upstream: <https://github.com/OpenHelix-Team/VLA-Adapter>
- Licence: MIT.
- Copyright: `Copyright (c) 2025 Yihao Wang, Pengxiang Ding, Lingxiao Li,
  Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao,
  Pengxu Hou, Siteng Huang, Yifan Tang, Wenhui Wang, Ru Zhang, Jianyi Liu,
  and Donglin Wang.`
- Files here: the resize and quaternion helpers in
  `benchmarks/x7_openvla_server.py`, from
  `experiments/robot/libero/libero_utils.py`.
- Source:
  <https://raw.githubusercontent.com/OpenHelix-Team/VLA-Adapter/23fa0c9c159e2aa04341cdd3e924f44061311060/LICENSE>
- Licence text: [MIT, VLA-Adapter](#mit-vla-adapter).

`benchmarks/OPENVLA_LICENSE.txt` carries the notices of these four
policy-code sources beside the scripts that use them.

## three.js

- Version: 0.185.1 (release r185), npm package `three`.
- Upstream: <https://github.com/mrdoob/three.js>
- Licence: MIT.
- Copyright: `Copyright © 2010-2026 three.js authors`
- Files here: `three.module.min.js`, `three.core.min.js` and
  `OrbitControls.js` in `src/specola/dashboard/static/vendor/`. Each is
  byte-identical to the npm package's file (SHA-256 values in
  `vendor/NOTICE.txt`). `OrbitControls.js` has no header of its own,
  upstream or here.
- Source: <https://unpkg.com/three@0.185.1/LICENSE>, identical to
  `LICENSE` at git tag `r185`.
- Licence text: [MIT, three.js](#mit-threejs).

## Socket.IO client

- Version: 4.8.1, npm package `socket.io-client`.
- Upstream: <https://github.com/socketio/socket.io>
  (`packages/socket.io-client`)
- Licence: MIT.
- Copyright: `Copyright (c) 2014-present Guillermo Rauch and Socket.IO
  contributors`
- Files here: `src/specola/dashboard/static/vendor/socket.io.min.js`,
  byte-identical to <https://cdn.socket.io/4.8.1/socket.io.min.js>.
- Source: <https://unpkg.com/socket.io-client@4.8.1/LICENSE>, identical
  to `packages/socket.io-client/LICENSE` at git tag
  `socket.io-client@4.8.1`.
- Licence text: [MIT, Socket.IO](#mit-socketio).

## IBM Plex Sans

- Version: font version 3.201, Latin subset, as packaged in
  `@fontsource/ibm-plex-sans` 5.3.0 (a Google Fonts build).
- Upstream: <https://github.com/IBM/plex>
- Licence: SIL Open Font License 1.1.
- Copyright embedded in the font files (name ID 0): `Copyright 2019 IBM
  Corp. All rights reserved.`
- Copyright and Reserved Font Name stated by IBM's licence file:
  `Copyright © 2017 IBM Corp. with Reserved Font Name "Plex"`
- Files here: `ibm-plex-sans-400.woff2`, `ibm-plex-sans-500.woff2` and
  `ibm-plex-sans-600.woff2` in `src/specola/dashboard/static/vendor/fonts/`,
  byte-identical to the Latin files in `@fontsource/ibm-plex-sans` 5.3.0.
- Source: the fonts' name tables, read with `fontTools`;
  <https://cdn.jsdelivr.net/npm/@fontsource/ibm-plex-sans@5.3.0/LICENSE>;
  <https://raw.githubusercontent.com/IBM/plex/39276912b05f700ee1dd491950c35d8697214208/LICENSE.txt>
  (same text as `ofl/ibmplexsans/OFL.txt` in Google Fonts).
- Licence text: [SIL Open Font License 1.1](#sil-open-font-license-11).

## JetBrains Mono

- Version: font version 2.211, Latin subset, as packaged in
  `@fontsource/jetbrains-mono` 5.3.0 (a Google Fonts build).
- Upstream: <https://github.com/JetBrains/JetBrainsMono>
- Licence: SIL Open Font License 1.1.
- Copyright: `Copyright 2020 The JetBrains Mono Project Authors
  (https://github.com/JetBrains/JetBrainsMono)`, the same in the font
  files and in JetBrains' licence file.
- Reserved Font Name: none. JetBrains' licence file declares none.
- Files here: `jetbrains-mono-400.woff2` and `jetbrains-mono-500.woff2` in
  `src/specola/dashboard/static/vendor/fonts/`, byte-identical to the
  Latin files in `@fontsource/jetbrains-mono` 5.3.0.
- Source: the fonts' name tables, read with `fontTools`;
  <https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5.3.0/LICENSE>;
  <https://raw.githubusercontent.com/JetBrains/JetBrainsMono/19371302b95d218af43299bce79ddbddd0bc364d/OFL.txt>
- Licence text: [SIL Open Font License 1.1](#sil-open-font-license-11).

## Reviewed and not listed

`src/specola/data/mujoco-tls-3.3.1.json` quotes three one-line
declarations from the Python 3.11 C API (`PyCFunction_GetFunction`,
`PyCFunction_GetFlags` and the `PyCFunction` function-pointer type). They
are interface declarations, and this file gives no notice for them. That
is a judgement, not a legal finding.

## Licence texts

### Apache License 2.0

Covers the MuJoCo source excerpts. Text from MuJoCo's `LICENSE` at the
five tags above.

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

### MIT, robosuite

```text
MIT License

Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This software includes the partial implementation of Deepmind Mujoco https://github.com/deepmind/mujoco.
Deepmind Mujoco is licensed under the Apache License, Version 2.0 (the "License");
you may not use the files except in compliance with the License.

You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
```

### MIT, openvla-oft

```text
MIT License

Copyright (c) 2025 Moo Jin Kim, Chelsea Finn, Percy Liang.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MIT, OpenVLA

```text
MIT License

Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MIT, VLA-Adapter

```text
MIT License

Copyright (c) 2025 Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, Siteng Huang, Yifan Tang, Wenhui Wang, Ru Zhang, Jianyi Liu, and Donglin Wang.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MIT, three.js

```text
The MIT License

Copyright © 2010-2026 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### MIT, Socket.IO

```text
The MIT License (MIT)

Copyright (c) 2014-present Guillermo Rauch and Socket.IO contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### SIL Open Font License 1.1

Covers IBM Plex Sans and JetBrains Mono. The text after the copyright
lines is the one shipped in both `@fontsource` 5.3.0 packages. From the
`SIL OPEN FONT LICENSE` heading on, it is identical to the text in IBM's
and JetBrains' own licence files.

```text
Copyright 2019 IBM Corp. All rights reserved.
Copyright © 2017 IBM Corp. with Reserved Font Name "Plex"
Copyright 2020 The JetBrains Mono Project Authors (https://github.com/JetBrains/JetBrainsMono)

This Font Software is licensed under the SIL Open Font License, Version 1.1.
This license is copied below, and is also available with a FAQ at:
http://scripts.sil.org/OFL


-----------------------------------------------------------
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
-----------------------------------------------------------

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded,
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may
include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless explicit written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
```
