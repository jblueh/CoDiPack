/*
 * CoDiPack, a Code Differentiation Package
 *
 * Copyright (C) 2015-2022 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum, Johannes Blühdorn (SciComp, TU Kaiserslautern)
 *
 * This file is part of CoDiPack (http://www.scicomp.uni-kl.de/software/codi).
 *
 * CoDiPack is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * CoDiPack is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * For other licensing options please contact us.
 *
 * Authors:
 *  - SciComp, TU Kaiserslautern:
 *    - Max Sagebaum
 *    - Johannes Blühdorn
 *    - Former members:
 *      - Tim Albring
 */
#pragma once

#include "../../config.h"
#include "../../misc/macros.hpp"
#include "../data/dataInterface.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  /** Helper structure for reading and writing the fixed size statement data.
   *
   *  A call to readForward or readReverse will populate the member data.
   *
   *  A call to first reserve and then store will first reserve the data on the data stream and afterwards push the
   *  data on the stream.
   *
   *  @tparam T_EvalHandle  Evaluation handle of the tape.
   *  @tparam T_FixedSizeData  Data stream that stores the fixed data.
   */
  template<typename T_EvalHandle, typename T_FixedSizeData>
  struct StatementFixedDataEntry {
    public:

      using EvalHandle = CODI_DD(T_EvalHandle, void*);                ///< See StatementFixedDataEntry.
      using FixedSizeData = CODI_DD(T_FixedSizeData, DataInterface);  ///< See StatementFixedDataEntry.

      EvalHandle handle;                              ///< The currently read handle.
      Config::ArgumentSize numberOfPassiveArguments;  ///< The currently read number of passive arguments.

      /// See StatementFixedDataEntry.
      /// @return The new position of in values.
      CODI_INLINE size_t readForward(char const* const values, size_t pos) {
        numberOfPassiveArguments = *((Config::ArgumentSize const*)(&values[pos]));
        pos += sizeof(Config::ArgumentSize);
        handle = *((EvalHandle const*)(&values[pos]));
        pos += sizeof(EvalHandle);

        return pos;
      }

      /// See StatementFixedDataEntry.
      /// @return The new position in values.
      CODI_INLINE size_t readReverse(char const* const values, size_t pos) {
        pos -= sizeof(EvalHandle);
        handle = *((EvalHandle const*)(&values[pos]));
        pos -= sizeof(Config::ArgumentSize);
        numberOfPassiveArguments = *((Config::ArgumentSize const*)(&values[pos]));

        return pos;
      }

      /// Reserve the data for the fixed size data stream.
      CODI_INLINE static void reserve(FixedSizeData& data) {
        data.reserveItems(sizeof(Config::ArgumentSize) + sizeof(EvalHandle));
      }

      /// Store the data for the fixed data stream.
      CODI_INLINE static void store(FixedSizeData& data, Config::ArgumentSize numberOfPassiveArguments,
                                    EvalHandle handle) {
        data.pushData(numberOfPassiveArguments);
        data.pushData(handle);
      }
  };
}
