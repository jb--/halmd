if(HALMD_WITH_GPU)
  add_executable(test_unit_numeric_mp_dsfun
    dsfun.cpp
    dsfun_kernel.cu
  )
  target_link_libraries(test_unit_numeric_mp_dsfun
    ${HALMD_TEST_LIBRARIES}
  )
  add_test(unit/numeric/mp/dsfun
    test_unit_numeric_mp_dsfun --log_level=test_suite
  )
endif(HALMD_WITH_GPU)
