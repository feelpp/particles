###  TEMPLATE.txt.tpl; coding: utf-8 ---

#  Author(s): Christophe Prud'homme <christophe.prudhomme@feelpp.org>
#       Date: 2020-06-03
#
#  Copyright (C) 2020 Cemosis
#
# Distributed under the GPL(GNU Public License):
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
#
#
#add_executable(particles 
#    particles.cpp
#)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS unit_test_framework)

add_library(nelder_mead INTERFACE)
target_include_directories(nelder_mead INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/)
target_link_libraries(nelder_mead INTERFACE Eigen3::Eigen)

add_executable(sim sim_main.cpp)
target_include_directories(sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sim PRIVATE nelder_mead)

add_executable(sim2 main.cpp simulation.cpp)
target_include_directories(sim2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sim2 PRIVATE nelder_mead)
#

enable_testing()


add_executable(test_nelder_mead test_nelder_mead.cpp)
target_link_libraries(test_nelder_mead PRIVATE Boost::unit_test_framework  nelder_mead)
add_test(NAME NelderMeadTests COMMAND test_nelder_mead)


if (PROJECT_ENABLE_FEELPP)
feelpp_add_application(laplacian SRCS laplacian.cpp TESTS INSTALL )


if(FEELPP_TOOLBOXES_FOUND)
    feelpp_add_application(toolbox SRCS toolbox.cpp LINK_LIBRARIES Feelpp::feelpp_toolbox_electric_lib TESTS)
endif()

foreach(binding laplacian)
    if ( SKBUILD_PROJECT_NAME )
            python_add_library(_${binding} MODULE _${binding}.cpp WITH_SOABI)
            target_link_libraries(_${binding} PRIVATE pybind11::headers Feelpp::feelpp)
            target_compile_definitions(_${binding} PRIVATE VERSION_INFO=${PROJECT_VERSION})
            install(TARGETS _${binding} DESTINATION feelpp/project)
    else()
        feelpp_add_pymodule(${binding} SRCS _${binding}.cpp DESTINATION feelpp/project)
    endif()
endforeach()
endif( PROJECT_ENABLE_FEELPP)