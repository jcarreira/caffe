file(GLOB_RECURSE examples_srcs "${PROJECT_SOURCE_DIR}/examples/*.cpp")

foreach(source_file ${examples_srcs})
  # get file name
  get_filename_component(name ${source_file} NAME_WE)
    
  # get folder name
  get_filename_component(path ${source_file} PATH)
  get_filename_component(folder ${path} NAME_WE)

  MESSAGE( STATUS "ADDING EXECUTABLE: " ${name} )    
  add_executable(${name} ${source_file})
  MESSAGE( STATUS "LINKING : " ${Caffe_LINK} )    
  target_link_libraries(${name} ${Caffe_LINK})

  MESSAGE( STATUS "LINKING CIRRUS" ${S} )    
  target_link_libraries(${name}
          /data/joao/caffe/cirrus/src/client/libclient.a
          /data/joao/caffe/cirrus/src/utils/libutils.a
          /data/joao/caffe/cirrus/src/common/libcommon.a
          /data/joao/caffe/cirrus/src/server/libserver.a
          /data/joao/caffe/cirrus/src/cache_manager/libevictionpolicies.a)
  include_directories(/data/joao/caffe/cirrus/src/)
  include_directories(/data/joao/caffe/cirrus/third_party/flatbuffers/include/)
  include_directories(/data/joao/caffe/cirrus/third_party/libcuckoo)
  caffe_default_properties(${name})

  # set back RUNTIME_OUTPUT_DIRECTORY
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/examples/${folder}")
  set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS -std=c++14)

  caffe_set_solution_folder(${name} examples)

  # install
  install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})


  if(UNIX OR APPLE)
    # Funny command to make tutorials work
    # TODO: remove in future as soon as naming is standardized everywhere
    set(__outname ${PROJECT_BINARY_DIR}/examples/${folder}/${name}${Caffe_POSTFIX})
    add_custom_command(TARGET ${name} POST_BUILD
                       COMMAND ln -sf "${__outname}" "${__outname}.bin")
  endif()
endforeach()
